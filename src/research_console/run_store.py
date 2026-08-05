"""Append-only experiment records used by training and the research console."""

import json
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import ArtifactRef, BatchMetrics, EpochMetrics, RunMetadata, RunStatus, RunSummary
from .telemetry import SystemTelemetrySampler


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp_file:
        temp_file.write(serialized)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


class RunLogger:
    """Write a versioned run record with bounded, live-readable telemetry."""

    def __init__(self, runs_root: Path, metadata: RunMetadata, *, system_sample_interval_seconds: float = 5.0) -> None:
        if not RUN_ID_PATTERN.fullmatch(metadata.run_id):
            raise ValueError(f"Invalid run id: {metadata.run_id!r}")
        if system_sample_interval_seconds <= 0:
            raise ValueError("system_sample_interval_seconds must be positive.")
        self.run_dir = Path(runs_root) / metadata.run_id
        self._metadata = metadata
        self._system_sample_interval_seconds = system_sample_interval_seconds
        self._telemetry_sampler = SystemTelemetrySampler()
        self._last_system_sample_at: datetime | None = None
        self._last_gpu_memory_allocated_bytes: int | None = None
        self._last_gpu_memory_reserved_bytes: int | None = None
        self._best_epoch: int | None = None
        self._best_validation_mean_iou: float | None = None

    def _write_metadata(self) -> None:
        atomic_write_json(self.run_dir / "metadata.json", self._metadata.model_dump(mode="json", exclude_none=True))

    def start(self) -> None:
        if self.run_dir.exists() and (self.run_dir / "metadata.json").exists():
            raise FileExistsError(f"Run already exists: {self.run_dir}")
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self._metadata = self._metadata.model_copy(update={"status": RunStatus.RUNNING, "started_at": utc_now(), "ended_at": None})
        self._write_metadata()

    def maybe_log_system(self, *, force: bool = False, gpu_memory_allocated_bytes: int | None = None, gpu_memory_reserved_bytes: int | None = None) -> bool:
        if gpu_memory_allocated_bytes is not None:
            self._last_gpu_memory_allocated_bytes = gpu_memory_allocated_bytes
        if gpu_memory_reserved_bytes is not None:
            self._last_gpu_memory_reserved_bytes = gpu_memory_reserved_bytes
        now = utc_now()
        if not force and self._last_system_sample_at is not None and (now - self._last_system_sample_at).total_seconds() < self._system_sample_interval_seconds:
            return False
        append_jsonl(
            self.run_dir / "system_metrics.jsonl",
            self._telemetry_sampler.collect(
                gpu_memory_allocated_bytes=self._last_gpu_memory_allocated_bytes,
                gpu_memory_reserved_bytes=self._last_gpu_memory_reserved_bytes,
            ).model_dump(mode="json", exclude_none=True),
        )
        self._last_system_sample_at = now
        return True

    def log_batch(self, *, epoch: int, batch: int, total_batches: int, loss: float, smoothed_loss: float | None = None, throughput_samples_per_second: float | None = None, eta_seconds: float | None = None, gpu_memory_allocated_bytes: int | None = None, gpu_memory_reserved_bytes: int | None = None) -> None:
        append_jsonl(self.run_dir / "metrics.jsonl", BatchMetrics(timestamp=utc_now(), epoch=epoch, batch=batch, total_batches=total_batches, loss=loss, smoothed_loss=smoothed_loss, throughput_samples_per_second=throughput_samples_per_second, eta_seconds=eta_seconds).model_dump(mode="json", exclude_none=True))
        self.maybe_log_system(gpu_memory_allocated_bytes=gpu_memory_allocated_bytes, gpu_memory_reserved_bytes=gpu_memory_reserved_bytes)

    def log_training_diagnostic(self, *, event_type: str, **payload: Any) -> None:
        """Append structured training diagnostics to metrics.jsonl."""
        append_jsonl(
            self.run_dir / "metrics.jsonl",
            {
                "schema_version": 1,
                "event_type": event_type,
                "timestamp": utc_now().isoformat(),
                **payload,
            },
        )
        self.maybe_log_system()

    def log_epoch(self, event: EpochMetrics) -> None:
        append_jsonl(self.run_dir / "metrics.jsonl", event.model_dump(mode="json", exclude_none=True))
        if self._metadata.provenance.protocol.valid and self._metadata.provenance.split_role.value == "crowdsourced_validation" and event.mean_iou is not None and (self._best_validation_mean_iou is None or event.mean_iou > self._best_validation_mean_iou):
            self._best_epoch = event.epoch
            self._best_validation_mean_iou = event.mean_iou
        self.maybe_log_system(force=True)

    def finish(self, *, status: RunStatus = RunStatus.COMPLETED, failure_reason: str | None = None, traceback_artifact: ArtifactRef | None = None) -> RunSummary:
        if status not in {RunStatus.COMPLETED, RunStatus.INTERRUPTED, RunStatus.INVALID, RunStatus.FAILED}:
            raise ValueError("A terminal run status is required.")
        ended_at = utc_now()
        self._metadata = self._metadata.model_copy(update={"status": status, "ended_at": ended_at})
        self._write_metadata()
        duration_seconds = (ended_at - self._metadata.started_at).total_seconds() if self._metadata.started_at else None
        summary = RunSummary(run_id=self._metadata.run_id, status=status, started_at=self._metadata.started_at, ended_at=ended_at, duration_seconds=duration_seconds, best_epoch=self._best_epoch, best_validation_mean_iou=self._best_validation_mean_iou, protocol=self._metadata.provenance.protocol, failure_reason=failure_reason, traceback_artifact=traceback_artifact)
        atomic_write_json(self.run_dir / "summary.json", summary.model_dump(mode="json", exclude_none=True))
        return summary

    def fail(self, error: BaseException) -> RunSummary:
        traceback_path = self.run_dir / "artifacts" / "failure_traceback.txt"
        traceback_path.write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        artifact = ArtifactRef(path="artifacts/failure_traceback.txt", kind="traceback", description="Terminal run traceback")
        self._metadata = self._metadata.model_copy(
            update={"artifact_refs": [*self._metadata.artifact_refs, artifact]}
        )
        return self.finish(status=RunStatus.FAILED, failure_reason=str(error), traceback_artifact=artifact)
