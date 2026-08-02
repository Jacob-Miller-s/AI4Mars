"""Versioned, portable records for AI4Mars research runs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = 1
CLASS_NAMES = ("soil", "bedrock", "sand", "big_rock")


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    INVALID = "invalid"


class SplitRole(str, Enum):
    TRAINING = "training"
    CROWDSOURCED_VALIDATION = "crowdsourced_validation"
    SEALED_FINAL_EXPERT_TEST = "sealed_final_expert_test"


class ArtifactRef(BaseModel):
    """An artifact location that is always portable and rooted in its run."""

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
    input_resolution: tuple[int, int] | None = None


class TrainingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: str
    scheduler: str | None = None
    learning_rate: float | None = Field(default=None, ge=0)
    class_weights: list[float] | None = None
    loss: str
    batch_size: int | None = Field(default=None, ge=1)
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
    """The durable, immutable identity and configuration of an experiment."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    experiment_name: str
    hypothesis: str | None = None
    tags: list[str] = Field(default_factory=list)
    researcher_notes: str | None = None
    status: RunStatus = RunStatus.QUEUED
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
    pixel_accuracy: float | None = Field(default=None, ge=0, le=1)
    mean_iou: float | None = Field(default=None, ge=0, le=1)
    per_class: dict[str, ClassMetrics] = Field(default_factory=dict)
    learning_rate: float | None = Field(default=None, ge=0)
    epoch_duration_seconds: float | None = Field(default=None, ge=0)
    confusion_matrix: list[list[int]] | None = None
    checkpoint: ArtifactRef | None = None


class BatchMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    event_type: Literal["batch"] = "batch"
    timestamp: datetime
    epoch: int = Field(ge=1)
    batch: int = Field(ge=1)
    total_batches: int = Field(ge=1)
    loss: float
    smoothed_loss: float | None = None
    throughput_samples_per_second: float | None = Field(default=None, ge=0)
    eta_seconds: float | None = Field(default=None, ge=0)


class SystemMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    event_type: Literal["system"] = "system"
    timestamp: datetime
    cpu_percent: float | None = Field(default=None, ge=0, le=100)
    ram_percent: float | None = Field(default=None, ge=0, le=100)
    ram_used_bytes: int | None = Field(default=None, ge=0)
    disk_read_bytes: int | None = Field(default=None, ge=0)
    disk_write_bytes: int | None = Field(default=None, ge=0)
    gpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_temperature_celsius: float | None = Field(default=None, ge=0)
    gpu_memory_used_bytes: int | None = Field(default=None, ge=0)
    gpu_memory_total_bytes: int | None = Field(default=None, ge=0)
    gpu_memory_allocated_bytes: int | None = Field(default=None, ge=0)
    gpu_memory_reserved_bytes: int | None = Field(default=None, ge=0)
    gpu_available: bool = False


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