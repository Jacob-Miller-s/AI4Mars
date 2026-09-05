"""Notebook-facing interface for the AI4Mars semantic reproduction."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ai4mars.dataset import AI4MarsDataset
from ai4mars.foundation import sha256_file
from ai4mars.metrics import segmentation_confusion_matrix, segmentation_metrics_from_confusion_matrix
from ai4mars.paper_model import build_deeplabv3plus
from ai4mars.paper_train import load_and_validate_config


BASELINE_CHECKPOINT_EPOCH = 25
BASELINE_CHECKPOINT_SHA256 = "90e74a9071d9bfb180d80ab2bb1927f1ea83a74d7e0601750873c2547a5ddaa3"
BASELINE_CHECKPOINT_URL = (
    "https://github.com/mandevautospa/AI4Mars/releases/download/"
    "semantic-reproduction-v1/deeplabv3plus-tesla-p100-seed42-best-val-miou.pth"
)
BASELINE_ONBOARDING_METRIC_RANGES = {
    "pixel_accuracy": (0.85, 0.90),
    "mean_iou": (0.72, 0.78),
}


@dataclass(frozen=True)
class VerifiedCheckpoint:
    path: Path
    sha256: str
    epoch: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class OnboardingSample:
    source_id: str
    sequence_id: str
    image_path: Path
    mask_path: Path
    class_counts: dict[int, int]


@dataclass(frozen=True)
class SamplePrediction:
    source_id: str
    image: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class OnboardingReport:
    device: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    predictions: tuple[SamplePrediction, ...]
    metrics: dict[str, Any]


def acquire_frozen_checkpoint(
    destination: Path,
    *,
    source_url: str = BASELINE_CHECKPOINT_URL,
    expected_sha256: str = BASELINE_CHECKPOINT_SHA256,
    expected_epoch: int = BASELINE_CHECKPOINT_EPOCH,
) -> VerifiedCheckpoint:
    """Return a verified local checkpoint, downloading it atomically if absent."""
    destination = Path(destination)
    if destination.is_file():
        return verify_frozen_checkpoint(
            destination,
            expected_sha256=expected_sha256,
            expected_epoch=expected_epoch,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        urllib.request.urlretrieve(source_url, temporary_path)
        verified = verify_frozen_checkpoint(
            temporary_path,
            expected_sha256=expected_sha256,
            expected_epoch=expected_epoch,
        )
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return VerifiedCheckpoint(
        path=destination,
        sha256=verified.sha256,
        epoch=verified.epoch,
        payload=verified.payload,
    )


def verify_frozen_checkpoint(
    path: Path,
    *,
    expected_sha256: str = BASELINE_CHECKPOINT_SHA256,
    expected_epoch: int = BASELINE_CHECKPOINT_EPOCH,
) -> VerifiedCheckpoint:
    """Verify and load a frozen checkpoint before model construction."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    observed_sha256 = sha256_file(checkpoint_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}."
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")

    epoch = int(payload.get("epoch", 0))
    if epoch != expected_epoch:
        raise ValueError(f"Checkpoint epoch mismatch: expected {expected_epoch}, observed {epoch}.")

    return VerifiedCheckpoint(
        path=checkpoint_path,
        sha256=observed_sha256,
        epoch=epoch,
        payload=payload,
    )


def load_onboarding_samples(sample_root: Path) -> tuple[OnboardingSample, ...]:
    """Load the fixed attributed onboarding sample after verifying every file."""
    root = Path(sample_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported onboarding sample manifest schema.")
    if manifest.get("source_split") != "val":
        raise ValueError("Onboarding samples must come from the development validation split.")

    rows = manifest.get("samples")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("The onboarding sample must contain exactly eight examples.")

    samples: list[OnboardingSample] = []
    source_ids: set[str] = set()
    sequence_ids: set[str] = set()
    observed_classes: set[int] = set()
    for row in rows:
        source_id = str(row["source_id"])
        sequence_id = str(row["sequence_id"])
        if source_id in source_ids or sequence_id in sequence_ids:
            raise ValueError("Onboarding samples must have unique source and sequence identifiers.")
        source_ids.add(source_id)
        sequence_ids.add(sequence_id)

        image_path = root / row["image_file"]
        mask_path = root / row["mask_file"]
        for path, expected_hash in (
            (image_path, row["image_sha256"]),
            (mask_path, row["mask_sha256"]),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Onboarding sample file not found: {path}")
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                raise ValueError(f"Onboarding sample SHA-256 mismatch: {path}")

        class_counts = {int(label): int(count) for label, count in row["class_counts"].items()}
        observed_classes.update(label for label, count in class_counts.items() if label in range(4) and count > 0)
        samples.append(
            OnboardingSample(
                source_id=source_id,
                sequence_id=sequence_id,
                image_path=image_path,
                mask_path=mask_path,
                class_counts=class_counts,
            )
        )

    if observed_classes != set(range(4)):
        raise ValueError(f"Onboarding sample does not cover all four classes: {sorted(observed_classes)}")
    return tuple(samples)


def run_onboarding(
    *,
    config_path: Path,
    checkpoint_path: Path,
    sample_root: Path,
    device: str = "auto",
    expected_metric_ranges: Mapping[str, tuple[float, float]] = BASELINE_ONBOARDING_METRIC_RANGES,
) -> OnboardingReport:
    """Run the complete frozen-checkpoint onboarding workflow."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    selected_device = torch.device(
        "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    config = load_and_validate_config(Path(config_path), enforce_training_batch_size=False)
    spec = config["paper_model_spec"]
    verified_checkpoint = verify_frozen_checkpoint(Path(checkpoint_path))
    samples = load_onboarding_samples(Path(sample_root))

    model = build_deeplabv3plus(spec, load_pretrained_encoder=False).to(selected_device)
    model.load_state_dict(verified_checkpoint.payload["model_state_dict"])
    model.eval()

    dataset = AI4MarsDataset(
        [(sample.image_path, sample.mask_path) for sample in samples],
        image_size=spec.input_size,
        require_original_shape_match=True,
        normalization_mean=spec.normalization_mean,
        normalization_std=spec.normalization_std,
    )
    mean = torch.tensor(spec.normalization_mean).view(3, 1, 1)
    std = torch.tensor(spec.normalization_std).view(3, 1, 1)
    aggregate_confusion = torch.zeros((spec.num_classes, spec.num_classes), dtype=torch.long)
    predictions: list[SamplePrediction] = []

    with torch.inference_mode():
        for sample, (image, target) in zip(samples, dataset):
            logits = model(image.unsqueeze(0).to(selected_device))
            expected_shape = (1, spec.num_classes, *target.shape)
            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}, expected {expected_shape}.")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Model produced non-finite logits for {sample.source_id}.")

            prediction = logits.argmax(dim=1).cpu().squeeze(0)
            observed_ids = set(prediction.unique().tolist())
            if not observed_ids.issubset(set(range(spec.num_classes))):
                raise RuntimeError(f"Model produced invalid class IDs for {sample.source_id}: {sorted(observed_ids)}")

            confusion = segmentation_confusion_matrix(
                prediction,
                target,
                num_classes=spec.num_classes,
                ignore_index=spec.ignore_index,
            )
            aggregate_confusion += confusion
            predictions.append(
                SamplePrediction(
                    source_id=sample.source_id,
                    image=(image * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).numpy(),
                    target=target.numpy(),
                    prediction=prediction.numpy(),
                    metrics=segmentation_metrics_from_confusion_matrix(confusion),
                )
            )

    metrics = segmentation_metrics_from_confusion_matrix(aggregate_confusion)
    for metric_name, bounds in expected_metric_ranges.items():
        observed = float(metrics[metric_name])
        lower, upper = bounds
        if not lower <= observed <= upper:
            raise RuntimeError(
                f"Onboarding {metric_name}={observed:.6f} is outside the expected range [{lower}, {upper}]."
            )

    return OnboardingReport(
        device=selected_device.type,
        checkpoint_sha256=verified_checkpoint.sha256,
        checkpoint_epoch=verified_checkpoint.epoch,
        predictions=tuple(predictions),
        metrics=metrics,
    )


def run_full_reproduction(
    *,
    config_path: Path,
    dataset_root: Path,
    output_root: Path,
    manifest_root: Path | None = None,
    run_id: str | None = None,
    resume_checkpoint: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Run the configured training workflow behind the notebook interface."""
    from ai4mars.paper_train import run_training

    args = Namespace(
        config=str(config_path),
        dataset_root=str(dataset_root),
        manifest_root=str(manifest_root) if manifest_root else None,
        output_root=str(output_root),
        run_id=run_id,
        architecture=None,
        backbone=None,
        pretrained_weights=None,
        optimizer=None,
        scheduler=None,
        class_weighting=None,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        seed=None,
        resolution=None,
        batch_size=None,
        gradient_accumulation=None,
        epochs=None,
        weight_decay=None,
        learning_rate=None,
        ignore_index=None,
        num_workers=None,
        checkpoint_interval=None,
        validation_interval=None,
        early_stopping=None,
        device=device,
        amp=None,
        validate_only=False,
        validation_level="metadata",
        zero_valid_audit_level=None,
    )
    result = run_training(args)
    if not isinstance(result, Path):
        raise RuntimeError("Training completed without producing a run directory.")
    return result


def run_sealed_expert_evaluation(
    *,
    config_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    output_root: Path,
    manifest_root: Path | None = None,
    run_id: str | None = None,
    device: str = "auto",
    splits: tuple[str, ...] = ("expert_min1", "expert_min2", "expert_min3"),
) -> Path:
    """Evaluate one frozen checkpoint against explicitly requested expert splits."""
    from ai4mars.paper_evaluate import run_expert_evaluation

    args = Namespace(
        config=str(config_path),
        checkpoint=str(checkpoint_path),
        splits=list(splits),
        dataset_root=str(dataset_root),
        manifest_root=str(manifest_root) if manifest_root else None,
        output_root=str(output_root),
        run_id=run_id,
        device=device,
        validation_level="metadata",
    )
    return run_expert_evaluation(args)