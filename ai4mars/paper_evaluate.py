"""Frozen-checkpoint expert evaluation, invoked by ``python -m ai4mars.paper_evaluate``.

This module is the ONLY place the sealed expert (min1/min2/min3 agreement) label
splits are ever scored. It loads a single frozen checkpoint produced by
``ai4mars.paper_train`` and reports metrics against the requested expert splits.
Nothing here ever selects a checkpoint, drives early stopping, or otherwise
feeds back into training.

Note on the row-normalized confusion matrix reported below: each row is divided
by its own ground-truth support, so the diagonal equals per-class RECALL
(true_positive / support), not IoU. IoU divides by the union of prediction and
ground truth and is reported separately in ``per_class``.
"""

from __future__ import annotations

import argparse
import csv
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai4mars.dataset import AI4MarsDataset, load_pairs_from_manifest
from ai4mars.foundation import current_git_commit, sha256_file
from ai4mars.paper_model import build_deeplabv3plus, paper_padding_metadata
from ai4mars.paper_reproduction import CLASS_NAMES, assert_no_reproduction_leakage, validate_manifest_files
from ai4mars.paper_train import load_and_validate_config
from ai4mars.records import (
    ClassMetrics,
    EnvironmentRecord,
    EpochMetrics,
    ModelRecord,
    ProtocolRecord,
    ProvenanceRecord,
    RunMetadata,
    RunStatus,
    SplitRole,
    ScientificRun,
    TrainingRecord,
    atomic_write_json,
)
from ai4mars.runtime import resolve_runtime_paths
from ai4mars.train_utils import evaluate

NORMALIZED_CONFUSION_MATRIX_NOTE = (
    "normalized_confusion_matrix is row-normalized by ground-truth support; "
    "its diagonal equals per-class recall, not IoU. See per_class[*].iou for IoU."
)

SPLIT_MANIFEST_KEYS = {
    "expert_min1": "expert_min1_manifest",
    "expert_min2": "expert_min2_manifest",
    "expert_min3": "expert_min3_manifest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", required=True, choices=sorted(SPLIT_MANIFEST_KEYS))
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest-root")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--validation-level", choices=("metadata", "full"), default="metadata")
    return parser.parse_args()


def load_frozen_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    """Restore model weights only. No optimizer, scheduler, or RNG state is touched."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint is missing model_state_dict: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    provenance = dict(checkpoint.get("metadata") or {})
    provenance["source_epoch"] = int(checkpoint.get("epoch", 0))
    return provenance


def _write_per_class_csv(path: Path, per_class: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["class_name", "support", "predicted", "true_positive", "false_positive", "false_negative", "iou", "dice_f1", "precision", "recall"]
        )
        for class_name, row in zip(CLASS_NAMES, per_class):
            writer.writerow(
                [class_name, row["support"], row["predicted"], row["true_positive"], row["false_positive"], row["false_negative"], row["iou"], row["dice_f1"], row["precision"], row["recall"]]
            )


def _write_confusion_matrix_csv(path: Path, matrix: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ground_truth \\ predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def run_expert_evaluation(args: argparse.Namespace) -> Path:
    config = load_and_validate_config(Path(args.config), enforce_training_batch_size=False)
    runtime, data, training, spec = config["runtime"], config["data"], config["training"], config["paper_model_spec"]
    paths = resolve_runtime_paths(
        project_root=Path(__file__).parent.parent,
        dataset_root=args.dataset_root or runtime.get("dataset_root"),
        manifest_root=args.manifest_root or runtime.get("manifest_root"),
        output_root=args.output_root or runtime.get("output_root"),
        run_id=args.run_id,
        accelerator=args.device or runtime.get("accelerator"),
    )
    paths.ensure_writable_roots()
    device = torch.device("cuda" if paths.accelerator == "cuda" or (paths.accelerator == "auto" and torch.cuda.is_available()) else "cpu")
    if paths.accelerator == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")

    missing = sorted(name for name in args.splits if not data.get(SPLIT_MANIFEST_KEYS[name]))
    if missing:
        raise ValueError(f"Configuration is missing manifest paths for requested expert splits: {missing}")

    expert_manifests = {name: paths.manifest_root / data[SPLIT_MANIFEST_KEYS[name]] for name in args.splits}
    train_val_manifests = {"train": paths.manifest_root / data["train_manifest"], "val": paths.manifest_root / data["val_manifest"]}
    audit_manifests = {**train_val_manifests, **expert_manifests}
    assert_no_reproduction_leakage(audit_manifests)
    if args.validation_level == "full":
        for name, path in expert_manifests.items():
            validate_manifest_files(path, paths.dataset_root, split_name=name)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_deeplabv3plus(spec).to(device)
    padding_metadata = paper_padding_metadata(spec)
    checkpoint_provenance = load_frozen_checkpoint(model, checkpoint_path, device)

    pairs = {
        name: load_pairs_from_manifest(path, dataset_root=paths.dataset_root, required_label_scheme="NAV", require_shape_match=True)
        for name, path in expert_manifests.items()
    }
    options = {
        "image_size": spec.input_size,
        "require_original_shape_match": True,
        "normalization_mean": spec.normalization_mean,
        "normalization_std": spec.normalization_std,
    }
    datasets = {name: AI4MarsDataset(value, **options) for name, value in pairs.items()}
    loader_options = {"batch_size": int(training["batch_size"]), "num_workers": int(training["num_workers"]), "pin_memory": device.type == "cuda"}
    loaders = {name: DataLoader(dataset, shuffle=False, **loader_options) for name, dataset in datasets.items()}

    loss_fn = nn.CrossEntropyLoss(ignore_index=spec.ignore_index)

    logger = ScientificRun(
        paths.event_root,
        RunMetadata(
            run_id=paths.run_id,
            experiment_name=f"{config.get('experiment_name', paths.run_id)}-expert-evaluation",
            paper_reproduction=True,
            tags=["paper-reproduction", "expert-evaluation", "sealed-final-test"],
            provenance=ProvenanceRecord(
                dataset_name="AI4Mars",
                dataset_version="ai4mars-dataset-merged-0.6",
                dataset_manifest_sha256=sha256_file(paths.manifest_root / data["dataset_manifest"]),
                split_manifest_hashes={name: sha256_file(path) for name, path in audit_manifests.items()},
                split_role=SplitRole.SEALED_FINAL_EXPERT_TEST,
                protocol=ProtocolRecord(
                    valid=True,
                    notes=[
                        "Sealed expert evaluation of a frozen checkpoint. Never used for checkpoint "
                        "selection, early stopping, or any other training decision.",
                        f"checkpoint_source={checkpoint_path}",
                    ],
                ),
                git_commit=current_git_commit(paths.project_root),
            ),
            model=ModelRecord(
                name=spec.architecture,
                encoder=spec.backbone,
                pretrained_weights=spec.pretrained_weights,
                parameter_count=sum(item.numel() for item in model.parameters()),
                input_resolution=spec.input_size,
                requested_input_size=tuple(padding_metadata["requested_input_size"]),
                internal_padding_multiple=padding_metadata["internal_padding_multiple"],
                internal_padded_size_for_513=tuple(padding_metadata["internal_padded_size_for_513"]),
                input_padding_policy=padding_metadata["input_padding_policy"],
                input_padding_mode=padding_metadata["input_padding_mode"],
                normalized_padding_value=padding_metadata["normalized_padding_value"],
                output_crop_policy=padding_metadata["output_crop_policy"],
            ),
            training=TrainingRecord(optimizer=training.get("optimizer", "none"), scheduler=training.get("scheduler"), loss="CrossEntropyLoss", batch_size=int(training["batch_size"])),
            environment=EnvironmentRecord(python=platform.python_version(), pytorch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None),
        ),
    )
    logger.start()

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_provenance": checkpoint_provenance,
        "note": NORMALIZED_CONFUSION_MATRIX_NOTE,
        "splits": {},
    }
    with torch.no_grad():
        for index, name in enumerate(args.splits, start=1):
            metrics = evaluate_split(model, loaders[name], loss_fn, device)
            report["splits"][name] = metrics
            _write_per_class_csv(logger.run_dir / "artifacts" / f"{name}_per_class_metrics.csv", metrics["per_class"])
            _write_confusion_matrix_csv(logger.run_dir / "artifacts" / f"{name}_confusion_matrix_raw.csv", metrics["confusion_matrix"])
            _write_confusion_matrix_csv(logger.run_dir / "artifacts" / f"{name}_confusion_matrix_normalized.csv", metrics["normalized_confusion_matrix"])
            per_class = {class_name: ClassMetrics(**{key: value for key, value in row.items() if key != "class_index"}) for class_name, row in zip(CLASS_NAMES, metrics["per_class"])}
            logger.log_epoch(
                EpochMetrics(
                    timestamp=datetime.now(timezone.utc),
                    epoch=index,
                    evaluation_split=name,
                    val_loss=metrics["val_loss"],
                    pixel_accuracy=metrics["pixel_accuracy"],
                    mean_iou=metrics["mean_iou"],
                    per_class=per_class,
                    confusion_matrix=metrics["confusion_matrix"],
                )
            )

    atomic_write_json(logger.run_dir / "artifacts" / "expert_evaluation.json", report)
    logger.finish(status=RunStatus.COMPLETED)
    return logger.run_dir


def main() -> None:
    run_expert_evaluation(parse_args())


def evaluate_split(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict[str, Any]:
    return evaluate(model, loader, loss_fn, device, return_detailed_metrics=True)


if __name__ == "__main__":
    main()
