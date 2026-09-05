"""Minimal, portable scientific records for AI4Mars reproduction runs."""

import json
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = 1


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SplitRole(str, Enum):
    TRAINING = "training"
    CROWDSOURCED_VALIDATION = "crowdsourced_validation"
    SEALED_FINAL_EXPERT_TEST = "sealed_final_expert_test"


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    kind: str
    description: str | None = None

    @field_validator("path")
    @classmethod
    def require_portable_relative_path(cls, value: str) -> str:
        if not value or "\\" in value:
            raise ValueError("Artifact paths must be non-empty POSIX relative paths.")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("Artifact paths must remain inside the run directory.")
        if ":" in path.parts[0]:
            raise ValueError("Artifact paths must not include a drive prefix.")
        return path.as_posix()


class ProtocolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    failed_gates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_version: str
    source_record: str | None = None
    dataset_manifest_sha256: str
    split_manifest_hashes: dict[str, str]
    split_role: SplitRole
    protocol: ProtocolRecord
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    random_seeds: dict[str, int] = Field(default_factory=dict)
    determinism: dict[str, Any] = Field(default_factory=dict)


class ModelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    encoder: str | None = None
    pretrained_weights: str | None = None
    parameter_count: int | None = Field(default=None, ge=0)
    # The scientifically reported model input resolution (e.g. 513x513 for the
    # paper reproduction). This is the only field consumers should treat as the
    # experimental image resolution.
    input_resolution: tuple[int, int] | None = None
    # The following fields describe an internal, non-scientific implementation
    # detail: some encoders require spatial dimensions divisible by their
    # output stride, so PaperAlignedDeepLabV3Plus pads/crops around the
    # unmodified input_resolution above. None of these fields should ever be
    # read as the experimental input resolution.
    requested_input_size: tuple[int, int] | None = None
    internal_padding_multiple: int | None = Field(default=None, ge=1)
    internal_padded_size_for_513: tuple[int, int] | None = None
    input_padding_policy: str | None = None
    input_padding_mode: str | None = None
    normalized_padding_value: float | None = None
    output_crop_policy: str | None = None


class TrainingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: str
    scheduler: str | None = None
    learning_rate: float | None = Field(default=None, ge=0)
    class_weights: list[float] | None = None
    loss: str
    batch_size: int | None = Field(default=None, ge=1)
    physical_batch_size: int | None = Field(default=None, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    effective_batch_size: int | None = Field(default=None, ge=1)
    class_weighting_strategy: str | None = None
    epochs: int | None = Field(default=None, ge=1)
    augmentation: dict[str, Any] = Field(default_factory=dict)
    precision_mode: str | None = None


class EnvironmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str | None = None
    pytorch: str | None = None
    cuda: str | None = None
    cudnn: str | None = None
    gpu: str | None = None
    cpu: str | None = None
    memory_total_bytes: int | None = Field(default=None, ge=0)


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    experiment_name: str
    experiment_family: str | None = None
    paper_reproduction: bool = False
    dataset_scope: str | None = None
    hypothesis: str | None = None
    tags: list[str] = Field(default_factory=list)
    researcher_notes: str | None = None
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime | None = None
    ended_at: datetime | None = None
    provenance: ProvenanceRecord
    model: ModelRecord
    training: TrainingRecord
    environment: EnvironmentRecord = Field(default_factory=EnvironmentRecord)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)


class ClassMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    iou: float | None = Field(default=None, ge=0, le=1)
    dice_f1: float | None = Field(default=None, ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    event_type: Literal["epoch"] = "epoch"
    timestamp: datetime
    epoch: int = Field(ge=1)
    train_loss: float | None = None
    val_loss: float | None = None
    unweighted_val_loss: float | None = None
    evaluation_split: str = "val"
    pixel_accuracy: float | None = Field(default=None, ge=0, le=1)
    mean_iou: float | None = Field(default=None, ge=0, le=1)
    per_class: dict[str, ClassMetrics] = Field(default_factory=dict)
    learning_rate: float | None = Field(default=None, ge=0)
    epoch_duration_seconds: float | None = Field(default=None, ge=0)
    confusion_matrix: list[list[int]] | None = None
    checkpoint: ArtifactRef | None = None


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str
    status: RunStatus
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    best_epoch: int | None = Field(default=None, ge=1)
    best_validation_mean_iou: float | None = Field(default=None, ge=0, le=1)
    protocol: ProtocolRecord
    failure_reason: str | None = None
    traceback_artifact: ArtifactRef | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)


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


class ScientificRun:
    """Persist configuration, provenance, epoch metrics, and a final summary."""

    def __init__(self, runs_root: Path, metadata: RunMetadata) -> None:
        if not RUN_ID_PATTERN.fullmatch(metadata.run_id):
            raise ValueError(f"Invalid run id: {metadata.run_id!r}")
        self.run_dir = Path(runs_root) / metadata.run_id
        self._metadata = metadata
        self._best_epoch: int | None = None
        self._best_validation_mean_iou: float | None = None

    def _write_metadata(self) -> None:
        atomic_write_json(self.run_dir / "metadata.json", self._metadata.model_dump(mode="json", exclude_none=True))

    def start(self) -> None:
        if self.run_dir.exists() and (self.run_dir / "metadata.json").exists():
            raise FileExistsError(f"Run already exists: {self.run_dir}")
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self._metadata = self._metadata.model_copy(
            update={"status": RunStatus.RUNNING, "started_at": utc_now(), "ended_at": None}
        )
        self._write_metadata()

    def log_epoch(self, event: EpochMetrics) -> None:
        append_jsonl(self.run_dir / "metrics.jsonl", event.model_dump(mode="json", exclude_none=True))
        is_validation = self._metadata.provenance.split_role == SplitRole.CROWDSOURCED_VALIDATION
        if is_validation and event.mean_iou is not None and (
            self._best_validation_mean_iou is None or event.mean_iou > self._best_validation_mean_iou
        ):
            self._best_epoch = event.epoch
            self._best_validation_mean_iou = event.mean_iou

    def finish(
        self,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        failure_reason: str | None = None,
        traceback_artifact: ArtifactRef | None = None,
    ) -> RunSummary:
        if status not in {RunStatus.COMPLETED, RunStatus.FAILED}:
            raise ValueError("A terminal run status is required.")
        ended_at = utc_now()
        self._metadata = self._metadata.model_copy(update={"status": status, "ended_at": ended_at})
        self._write_metadata()
        duration_seconds = (
            (ended_at - self._metadata.started_at).total_seconds() if self._metadata.started_at else None
        )
        summary = RunSummary(
            run_id=self._metadata.run_id,
            status=status,
            started_at=self._metadata.started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            best_epoch=self._best_epoch,
            best_validation_mean_iou=self._best_validation_mean_iou,
            protocol=self._metadata.provenance.protocol,
            failure_reason=failure_reason,
            traceback_artifact=traceback_artifact,
        )
        atomic_write_json(self.run_dir / "summary.json", summary.model_dump(mode="json", exclude_none=True))
        return summary

    def fail(self, error: BaseException) -> RunSummary:
        traceback_path = self.run_dir / "artifacts" / "failure_traceback.txt"
        traceback_path.write_text("".join(traceback.format_exception(error)), encoding="utf-8")
        artifact = ArtifactRef(
            path="artifacts/failure_traceback.txt",
            kind="traceback",
            description="Terminal run traceback",
        )
        self._metadata = self._metadata.model_copy(
            update={"artifact_refs": [*self._metadata.artifact_refs, artifact]}
        )
        return self.finish(
            status=RunStatus.FAILED,
            failure_reason=str(error),
            traceback_artifact=artifact,
        )