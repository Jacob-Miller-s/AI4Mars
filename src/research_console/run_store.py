"""Append-only experiment records used by training and the research console."""

from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    ArtifactRef,
    BatchMetrics,
    EpochMetrics,
    RunMetadata,
    RunStatus,
    RunSummary,
    SystemMetrics,
)
from .telemetry import SystemTelemetrySampler


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_file.write(serialized)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and flush one durable JSONL event."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(line)
        output.flush()
        os.fsync(output.fileno())


def read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    """Read valid JSONL events, ignoring only an incomplete final line."""
    path = Path(path)
    if not path.exists():
        return []
    raw_lines = path.read_bytes().splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        is_final_line = index == len(raw_lines) - 1
        try:
            decoded = json.loads(raw_line)
        except json.JSONDecodeError:
            if is_final_line and not raw_line.endswith((b"\n", b"\r")):
                break
            raise ValueError(f"Malformed JSONL event in {path} at line {index + 1}.") from None
        if not isinstance(decoded, dict):
            raise ValueError(f"JSONL event in {path} at line {index + 1} must be an object.")
        records.append(decoded)
    return records


class RunReader:
    """Read a durable run directory without requiring an in-memory producer."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def system_metrics_path(self) -> Path:
        return self.run_dir / "system_metrics.jsonl"

    def metadata(self) -> RunMetadata:
        return RunMetadata.model_validate_json(self.metadata_path.read_text(encoding="utf-8"))

    def summary(self) -> RunSummary | None:
        if not self.summary_path.exists():
            return None
        return RunSummary.model_validate_json(self.summary_path.read_text(encoding="utf-8"))

    def metrics(self) -> list[dict[str, Any]]:
        return read_jsonl_tolerant(self.metrics_path)

    def system_metrics(self) -> list[dict[str, Any]]:
        return read_jsonl_tolerant(self.system_metrics_path)


class RunLogger:
    """Write a versioned run record with bounded, live-readable telemetry."""

    def __init__(
        self,
        runs_root: Path,
        metadata: RunMetadata,
        *,
        system_sample_interval_seconds: float = 5.0,
        telemetry_sampler: SystemTelemetrySampler | None = None,
    ) -> None:
        if not RUN_ID_PATTERN.fullmatch(metadata.run_id):
            raise ValueError(f"Invalid run id: {metadata.run_id!r}")
        if system_sample_interval_seconds <= 0:
            raise ValueError("system_sample_interval_seconds must be positive.")
        self.runs_root = Path(runs_root)
        self.run_dir = self.runs_root / metadata.run_id
        self._metadata = metadata
        self._system_sample_interval_seconds = system_sample_interval_seconds
        self._telemetry_sampler = telemetry_sampler or SystemTelemetrySampler()
        self._last_system_sample_at: datetime | None = None
        self._last_gpu_memory_allocated_bytes: int | None = None
        self._last_gpu_memory_reserved_bytes: int | None = None
        self._best_epoch: int | None = None
        self._best_validation_mean_iou: float | None = None

    @property
    def metadata(self) -> RunMetadata:
        return self._metadata

    @property
    def reader(self) -> RunReader:
        return RunReader(self.run_dir)

    def start(self) -> None:
        if self.run_dir.exists() and (self.run_dir / "metadata.json").exists():
            raise FileExistsError(f"Run already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self._metadata = self._metadata.model_copy(
            update={"status": RunStatus.RUNNING, "started_at": utc_now(), "ended_at": None}
        )
        atomic_write_json(
            self.reader.metadata_path,
            self._metadata.model_dump(mode="json", exclude_none=True),
        )

    def log_batch(
        self,
        *,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        smoothed_loss: float | None = None,
        throughput_samples_per_second: float | None = None,
        eta_seconds: float | None = None,
        gpu_memory_allocated_bytes: int | None = None,
        gpu_memory_reserved_bytes: int | None = None,
    ) -> None:
        event = BatchMetrics(
            timestamp=utc_now(),
            epoch=epoch,
            batch=batch,
            total_batches=total_batches,
            loss=loss,
            smoothed_loss=smoothed_loss,
            throughput_samples_per_second=throughput_samples_per_second,
            eta_seconds=eta_seconds,
        )
        append_jsonl(self.reader.metrics_path, event.model_dump(mode="json", exclude_none=True))
        self.maybe_log_system(
            gpu_memory_allocated_bytes=gpu_memory_allocated_bytes,
            gpu_memory_reserved_bytes=gpu_memory_reserved_bytes,
        )

    def log_epoch(self, event: EpochMetrics) -> None:
        append_jsonl(self.reader.metrics_path, event.model_dump(mode="json", exclude_none=True))
        if (
            self._metadata.provenance.protocol.valid
            and self._metadata.provenance.split_role.value == "crowdsourced_validation"
            and event.mean_iou is not None
            and (
                self._best_validation_mean_iou is None
                or event.mean_iou > self._best_validation_mean_iou
            )
        ):
            self._best_epoch = event.epoch
            self._best_validation_mean_iou = event.mean_iou
        self.maybe_log_system(force=True)

    def maybe_log_system(
        self,
        *,
        force: bool = False,
        gpu_memory_allocated_bytes: int | None = None,
        gpu_memory_reserved_bytes: int | None = None,
    ) -> bool:
        if gpu_memory_allocated_bytes is not None:
            self._last_gpu_memory_allocated_bytes = gpu_memory_allocated_bytes
        if gpu_memory_reserved_bytes is not None:
            self._last_gpu_memory_reserved_bytes = gpu_memory_reserved_bytes
        now = utc_now()
        if (
            not force
            and self._last_system_sample_at is not None
            and (now - self._last_system_sample_at).total_seconds()
            < self._system_sample_interval_seconds
        ):
            return False
        event = self._telemetry_sampler.collect(
            gpu_memory_allocated_bytes=self._last_gpu_memory_allocated_bytes,
            gpu_memory_reserved_bytes=self._last_gpu_memory_reserved_bytes,
        )
        append_jsonl(self.reader.system_metrics_path, event.model_dump(mode="json", exclude_none=True))
        self._last_system_sample_at = now
        return True

    def register_artifact(self, artifact: ArtifactRef) -> None:
        """Add a portable artifact reference while preserving metadata atomically."""
        if artifact in self._metadata.artifact_refs:
            return
        self._metadata = self._metadata.model_copy(
            update={"artifact_refs": [*self._metadata.artifact_refs, artifact]}
        )
        atomic_write_json(
            self.reader.metadata_path,
            self._metadata.model_dump(mode="json", exclude_none=True),
        )

    def finish(
        self,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        failure_reason: str | None = None,
        traceback_artifact: ArtifactRef | None = None,
    ) -> RunSummary:
        if status not in {RunStatus.COMPLETED, RunStatus.INTERRUPTED, RunStatus.INVALID, RunStatus.FAILED}:
            raise ValueError("A terminal run status is required.")
        ended_at = utc_now()
        self._metadata = self._metadata.model_copy(update={"status": status, "ended_at": ended_at})
        atomic_write_json(
            self.reader.metadata_path,
            self._metadata.model_dump(mode="json", exclude_none=True),
        )
        started_at = self._metadata.started_at
        duration_seconds = (ended_at - started_at).total_seconds() if started_at else None
        summary = RunSummary(
            run_id=self._metadata.run_id,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            best_epoch=self._best_epoch,
            best_validation_mean_iou=self._best_validation_mean_iou,
            protocol=self._metadata.provenance.protocol,
            failure_reason=failure_reason,
            traceback_artifact=traceback_artifact,
            artifacts=self._metadata.artifact_refs,
        )
        atomic_write_json(self.reader.summary_path, summary.model_dump(mode="json", exclude_none=True))
        return summary

    def fail(self, error: BaseException) -> RunSummary:
        traceback_path = self.run_dir / "artifacts" / "failure_traceback.txt"
        traceback_path.write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        traceback_artifact = ArtifactRef(
            path="artifacts/failure_traceback.txt",
            kind="traceback",
            description="Terminal run traceback",
        )
        self._metadata = self._metadata.model_copy(
            update={
                "artifact_refs": [
                    *self._metadata.artifact_refs,
                    traceback_artifact,
                ]
            }
        )
        return self.finish(
            status=RunStatus.FAILED,
            failure_reason=str(error),
            traceback_artifact=traceback_artifact,
        )


def iter_run_directories(runs_root: Path) -> Iterable[Path]:
    root = Path(runs_root)
    if not root.exists():
        return []
    return (path for path in sorted(root.iterdir()) if path.is_dir() and (path / "metadata.json").exists())