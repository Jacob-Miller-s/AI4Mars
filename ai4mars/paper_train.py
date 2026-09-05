"""Paper-aligned DeepLabv3+ training implementation, invoked by ``python -m ai4mars.paper_train``."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ai4mars.dataset import AI4MarsDataset, load_pairs_from_manifest, normalize_ai4mars_mask
from ai4mars.foundation import build_checkpoint_metadata, current_git_commit, sha256_file
from ai4mars.paper_model import DeepLabV3PlusSpec, build_deeplabv3plus, paper_padding_metadata, validate_deeplabv3plus_spec
from ai4mars.paper_reproduction import (
    CLASS_NAMES,
    assert_no_reproduction_leakage,
    compute_paper_class_composition,
    summarize_reproduction_manifests,
    validate_manifest_files,
    validate_reproduction_manifest,
)
from ai4mars.records import (
    ArtifactRef,
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
from ai4mars.train_utils import evaluate, load_training_checkpoint, save_checkpoint, train_one_epoch


WEIGHTING_STRATEGIES = {"paper_complement_composition", "inverse_frequency"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    for name in (
        "dataset-root",
        "manifest-root",
        "output-root",
        "run-id",
        "architecture",
        "backbone",
        "pretrained-weights",
        "optimizer",
        "scheduler",
        "class-weighting",
        "resume-checkpoint",
    ):
        parser.add_argument(f"--{name}")
    for name in (
        "seed",
        "resolution",
        "batch-size",
        "gradient-accumulation",
        "epochs",
        "weight-decay",
        "learning-rate",
        "ignore-index",
        "num-workers",
        "checkpoint-interval",
        "validation-interval",
        "early-stopping",
    ):
        parser.add_argument(f"--{name}", type=float if name in {"weight-decay", "learning-rate"} else int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-level", choices=("metadata", "full"), default="metadata")
    parser.add_argument("--zero-valid-audit-level", choices=("metadata", "full"))
    return parser.parse_args()


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {key!r} must be a mapping.")
    return value


def _override(section: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        section[key] = value


def load_and_validate_config(
    path: Path,
    args: argparse.Namespace | None = None,
    *,
    enforce_training_batch_size: bool = True,
) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    runtime, data, model, training, logging = (
        _section(config, key) for key in ("runtime", "data", "model", "training", "logging")
    )
    config.update(runtime=runtime, data=data, model=model, training=training, logging=logging)

    if args is not None:
        for key, value in (
            ("dataset_root", args.dataset_root),
            ("manifest_root", args.manifest_root),
            ("output_root", args.output_root),
            ("run_id", args.run_id),
            ("accelerator", args.device),
        ):
            _override(runtime, key, value)
        for key, value in (
            ("seed", args.seed),
            ("batch_size", args.batch_size),
            ("gradient_accumulation_steps", args.gradient_accumulation),
            ("epochs", args.epochs),
            ("optimizer", args.optimizer),
            ("scheduler", args.scheduler),
            ("weight_decay", args.weight_decay),
            ("learning_rate", args.learning_rate),
            ("ignore_index", args.ignore_index),
            ("num_workers", args.num_workers),
            ("checkpoint_interval", args.checkpoint_interval),
            ("validation_interval", args.validation_interval),
            ("early_stopping_patience", args.early_stopping),
            ("class_weighting", args.class_weighting),
            ("mixed_precision", args.amp),
            ("resume_checkpoint", args.resume_checkpoint),
            ("zero_valid_audit_level", args.zero_valid_audit_level),
        ):
            _override(training, key, value)
        for key, value in (
            ("architecture", args.architecture),
            ("backbone", args.backbone),
            ("pretrained_weights", args.pretrained_weights),
        ):
            _override(model, key, value)
        if args.resolution is not None:
            model["input_size"] = [args.resolution, args.resolution]

    spec = DeepLabV3PlusSpec(
        architecture=model.get("architecture", ""),
        backbone=model.get("backbone", ""),
        pretrained_weights=model.get("pretrained_weights", ""),
        output_stride=int(model.get("output_stride", 16)),
        input_size=tuple(model.get("input_size", ())),
        num_classes=int(model.get("num_classes", 4)),
        ignore_index=int(training.get("ignore_index", 255)),
    )
    validate_deeplabv3plus_spec(spec)

    required = {"dataset_manifest", "train_manifest", "val_manifest"}
    if missing := sorted(name for name in required if not data.get(name)):
        raise ValueError(f"Missing required reproduction manifests: {missing}")

    if training.get("class_weighting") not in WEIGHTING_STRATEGIES:
        raise ValueError(f"class_weighting must be one of {sorted(WEIGHTING_STRATEGIES)}.")
    if training.get("optimizer") not in {"adamw", "sgd"} or training.get("scheduler") not in {
        "none",
        "cosine",
        "plateau",
    }:
        raise ValueError("Unsupported optimizer or scheduler.")

    for name in (
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "checkpoint_interval",
        "validation_interval",
    ):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"training.{name} must be at least 1.")

    if enforce_training_batch_size and int(training.get("batch_size", 0)) < 2:
        raise ValueError(
            "training.batch_size must be at least 2 for canonical DeepLabV3Plus training. "
            "With train-mode BatchNorm, the ASPP global-pooling branch emits [N, C, 1, 1], "
            "so physical N=1 is invalid. Gradient accumulation changes effective batch size "
            "but does not change per-forward-pass BatchNorm statistics."
        )

    if training.get("max_samples_per_split") is not None and int(training["max_samples_per_split"]) < 1:
        raise ValueError("training.max_samples_per_split must be at least 1 when configured.")
    if float(training.get("learning_rate", 0)) <= 0 or float(training.get("weight_decay", 0)) < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative.")

    training.setdefault("zero_valid_audit_level", "metadata")
    if training["zero_valid_audit_level"] not in {"metadata", "full"}:
        raise ValueError("training.zero_valid_audit_level must be either 'metadata' or 'full'.")

    config["paper_model_spec"] = spec
    return config


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weights(train_manifest: Path, strategy: str) -> tuple[torch.Tensor, dict[str, Any]]:
    composition = compute_paper_class_composition(train_manifest)
    if strategy == "paper_complement_composition":
        values = composition.class_weights
    else:
        inverse = [1.0 / max(value, 1e-8) for value in composition.class_proportions]
        mean = sum(inverse) / len(inverse)
        values = tuple(value / mean for value in inverse)
    return torch.tensor(values, dtype=torch.float32), {
        "strategy": strategy,
        "manifest_sha256": composition.manifest_sha256,
        "pixel_counts": list(composition.pixel_counts),
        "class_proportions": list(composition.class_proportions),
        "class_weights": list(values),
        "ignore_pixel_count": composition.ignore_pixel_count,
    }


def _optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = {"lr": float(training["learning_rate"]), "weight_decay": float(training["weight_decay"])}
    if training["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)


def _scheduler(optimizer: torch.optim.Optimizer, training: dict[str, Any]):
    if training["scheduler"] == "none":
        return None
    if training["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")


def _build_dataloaders(
    datasets: dict[str, Dataset],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    """Build split-aware loaders for training/validation/evaluation."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    loaders: dict[str, DataLoader] = {}
    persistent_workers = num_workers > 0
    for name, dataset in datasets.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    return loaders


class _IndexedDataset(Dataset):
    """Attach stable source identifiers to dataset samples for diagnostics."""

    def __init__(self, dataset: AI4MarsDataset, sample_ids: list[str]) -> None:
        if len(dataset) != len(sample_ids):
            raise ValueError("sample_ids length must match dataset length.")
        self.dataset = dataset
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, mask = self.dataset[index]
        return image, mask, self.sample_ids[index]


def _counts_to_int_map(raw_counts: dict[str, Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for key, value in raw_counts.items():
        label = int(key)
        count = int(value)
        if count < 0:
            raise ValueError("per_class_pixel_counts_json cannot contain negative counts.")
        counts[label] = count
    return counts


def _hash_json(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _count_mask_pixels(mask_path: Path) -> dict[int, int]:
    with Image.open(mask_path) as mask_file:
        mask = np.asarray(mask_file, dtype=np.int64)
    normalized = normalize_ai4mars_mask(mask, mask_path)
    unique, counts = np.unique(normalized, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def _preflight_zero_valid_training_rows(
    train_manifest: Path,
    dataset_root: Path,
    *,
    full_disk_audit: bool,
) -> tuple[dict[str, Any], set[str], dict[str, str]]:
    """Find deterministic all-ignore training rows and build an audit artifact."""
    rows = validate_reproduction_manifest(train_manifest, split_name="train")
    excluded: list[dict[str, Any]] = []
    retained_identity_rows: list[dict[str, str]] = []
    disk_count_mismatches: list[dict[str, Any]] = []
    excluded_mask_paths: set[str] = set()
    source_id_by_mask_path: dict[str, str] = {}

    for row in rows:
        relative_mask_path = row["dataset_relative_mask_path"]
        source_id_by_mask_path[relative_mask_path] = row["stable_source_image_id"]
        counts = _counts_to_int_map(json.loads(row["per_class_pixel_counts_json"]))
        valid_source_pixels = sum(count for label, count in counts.items() if label in {0, 1, 2, 3})
        manifest_all_ignore = valid_source_pixels == 0

        if full_disk_audit:
            observed_counts = _count_mask_pixels(dataset_root / relative_mask_path)
            if observed_counts != counts:
                disk_count_mismatches.append(
                    {
                        "stable_source_image_id": row["stable_source_image_id"],
                        "dataset_relative_mask_path": relative_mask_path,
                        "manifest_counts": {str(label): count for label, count in sorted(counts.items())},
                        "observed_counts": {str(label): count for label, count in sorted(observed_counts.items())},
                    }
                )

        if manifest_all_ignore:
            excluded_mask_paths.add(relative_mask_path)
            excluded.append(
                {
                    "stable_source_image_id": row["stable_source_image_id"],
                    "dataset_relative_image_path": row["dataset_relative_image_path"],
                    "dataset_relative_mask_path": relative_mask_path,
                    "original_per_class_pixel_counts": {str(label): count for label, count in sorted(counts.items())},
                    "source_manifest_all_ignore": True,
                    "valid_source_pixel_count": valid_source_pixels,
                    "exclusion_reason": "zero_supervised_pixels_in_manifest_counts",
                }
            )
        else:
            retained_identity_rows.append(
                {
                    "stable_source_image_id": row["stable_source_image_id"],
                    "dataset_relative_image_path": row["dataset_relative_image_path"],
                    "dataset_relative_mask_path": relative_mask_path,
                }
            )

    exclusion_list_hash = _hash_json(excluded)
    retained_identity_hash = _hash_json(retained_identity_rows)
    source_manifest_hash = sha256_file(train_manifest)
    artifact = {
        "source_train_rows_total": len(rows),
        "excluded_zero_valid_rows": len(excluded),
        "retained_training_rows": len(retained_identity_rows),
        "source_manifest_sha256": source_manifest_hash,
        "full_disk_count_audit_performed": full_disk_audit,
        "disk_count_mismatch_rows": disk_count_mismatches,
        "excluded_rows": excluded,
        "exclusion_list_sha256": exclusion_list_hash,
        "retained_training_identity_sha256": retained_identity_hash,
        "audit_hash": _hash_json(
            {
                "source_train_rows_total": len(rows),
                "excluded_zero_valid_rows": len(excluded),
                "retained_training_rows": len(retained_identity_rows),
                "source_manifest_sha256": source_manifest_hash,
                "exclusion_list_sha256": exclusion_list_hash,
                "retained_training_identity_sha256": retained_identity_hash,
            }
        ),
    }
    return artifact, excluded_mask_paths, source_id_by_mask_path


def run_training(args: argparse.Namespace) -> Path | dict[str, dict[str, int]]:
    config = load_and_validate_config(Path(args.config), args)
    runtime = config["runtime"]
    data = config["data"]
    training = config["training"]
    spec = config["paper_model_spec"]

    paths = resolve_runtime_paths(
        project_root=Path(__file__).parent.parent,
        dataset_root=runtime.get("dataset_root"),
        manifest_root=runtime.get("manifest_root"),
        output_root=runtime.get("output_root"),
        run_id=runtime.get("run_id"),
        accelerator=runtime.get("accelerator"),
    )
    paths.ensure_writable_roots()

    device = torch.device(
        "cuda"
        if paths.accelerator == "cuda" or (paths.accelerator == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if paths.accelerator == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")
    if training["mixed_precision"] and device.type != "cuda":
        raise ValueError("mixed_precision requires CUDA.")

    manifests = {
        "train": paths.manifest_root / data["train_manifest"],
        "val": paths.manifest_root / data["val_manifest"],
    }
    expert_manifest_keys = {
        "expert_min1": "expert_min1_manifest",
        "expert_min2": "expert_min2_manifest",
        "expert_min3": "expert_min3_manifest",
    }
    expert_manifests = {
        name: paths.manifest_root / data[key]
        for name, key in expert_manifest_keys.items()
        if data.get(key)
    }
    audit_manifests = {**manifests, **expert_manifests}
    assert_no_reproduction_leakage(audit_manifests)

    if args.validate_only and args.validation_level == "full":
        for name, path in audit_manifests.items():
            validate_manifest_files(path, paths.dataset_root, split_name=name)

    audit = summarize_reproduction_manifests(audit_manifests)
    if args.validate_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return audit

    _seed(int(training["seed"]))

    zero_valid_audit, excluded_mask_paths, source_id_by_mask_path = _preflight_zero_valid_training_rows(
        manifests["train"],
        paths.dataset_root,
        full_disk_audit=(training["zero_valid_audit_level"] == "full"),
    )

    train_pairs_all = load_pairs_from_manifest(
        manifests["train"],
        dataset_root=paths.dataset_root,
        required_label_scheme="NAV",
        require_shape_match=True,
    )
    train_pairs: list[tuple[Path, Path]] = []
    train_sample_ids: list[str] = []
    for image_path, mask_path in train_pairs_all:
        relative_mask_path = Path(mask_path).relative_to(paths.dataset_root).as_posix()
        if relative_mask_path in excluded_mask_paths:
            continue
        train_pairs.append((image_path, mask_path))
        train_sample_ids.append(source_id_by_mask_path.get(relative_mask_path, relative_mask_path))

    if not train_pairs:
        raise RuntimeError("Zero-valid preflight filtering removed every training row; no supervised training data remains.")

    val_pairs = load_pairs_from_manifest(
        manifests["val"],
        dataset_root=paths.dataset_root,
        required_label_scheme="NAV",
        require_shape_match=True,
    )
    pairs = {"train": train_pairs, "val": val_pairs}

    if training.get("max_samples_per_split") is not None:
        sample_limit = int(training["max_samples_per_split"])
        pairs = {name: values[:sample_limit] for name, values in pairs.items()}
        train_sample_ids = train_sample_ids[: len(pairs["train"])]

    options = {
        "image_size": spec.input_size,
        "require_original_shape_match": True,
        "normalization_mean": spec.normalization_mean,
        "normalization_std": spec.normalization_std,
    }
    datasets: dict[str, Dataset] = {name: AI4MarsDataset(value, **options) for name, value in pairs.items()}
    datasets["train"] = _IndexedDataset(datasets["train"], train_sample_ids)
    loaders = _build_dataloaders(
        datasets,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_deeplabv3plus(spec).to(device)
    weights, weighting = _weights(manifests["train"], training["class_weighting"])
    weights = weights.to(device)
    weighted_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=spec.ignore_index)
    unweighted_loss = nn.CrossEntropyLoss(ignore_index=spec.ignore_index)
    optimizer = _optimizer(model, training)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["mixed_precision"]))
    scheduler = _scheduler(optimizer, training)

    physical_batch_size = int(training["batch_size"])
    gradient_accumulation_steps = int(training["gradient_accumulation_steps"])
    padding_metadata = paper_padding_metadata(spec)
    metadata = build_checkpoint_metadata(
        project_root=paths.project_root,
        dataset_manifest_path=paths.manifest_root / data["dataset_manifest"],
        split_manifest_paths=audit_manifests,
        active_split_name="val",
        preprocessing={
            "input_size": list(spec.input_size),
            "normalization_mean": list(spec.normalization_mean),
            "normalization_std": list(spec.normalization_std),
            "mask_interpolation": "nearest",
            **padding_metadata,
        },
        loss_name="CrossEntropyLoss",
        loss_weights=weights.tolist(),
        model_name=f"{spec.architecture}/{spec.backbone}",
        seed=int(training["seed"]),
    )
    metadata["training_zero_valid_filter"] = {
        "enabled": True,
        "source_train_rows_total": zero_valid_audit["source_train_rows_total"],
        "excluded_zero_valid_rows": zero_valid_audit["excluded_zero_valid_rows"],
        "retained_training_rows": zero_valid_audit["retained_training_rows"],
        "source_manifest_sha256": zero_valid_audit["source_manifest_sha256"],
        "exclusion_list_sha256": zero_valid_audit["exclusion_list_sha256"],
        "retained_training_identity_sha256": zero_valid_audit["retained_training_identity_sha256"],
        "audit_hash": zero_valid_audit["audit_hash"],
        "zero_valid_audit_level": training["zero_valid_audit_level"],
    }

    snapshot = config | {"paper_model_spec": spec.metadata()}
    metadata.update(
        {
            "paper_reproduction": True,
            "model": spec.metadata(),
            "class_weighting": weighting,
            "configuration": snapshot,
        }
    )

    logger = ScientificRun(
        paths.event_root,
        RunMetadata(
            run_id=paths.run_id,
            experiment_name=config.get("experiment_name", paths.run_id),
            paper_reproduction=True,
            tags=["paper-reproduction", "msl", "navcam", "deeplabv3plus"],
            provenance=ProvenanceRecord(
                dataset_name="AI4Mars",
                dataset_version="ai4mars-dataset-merged-0.6",
                dataset_manifest_sha256=sha256_file(paths.manifest_root / data["dataset_manifest"]),
                split_manifest_hashes={name: sha256_file(path) for name, path in audit_manifests.items()},
                split_role=SplitRole.CROWDSOURCED_VALIDATION,
                protocol=ProtocolRecord(
                    valid=True,
                    notes=[
                        "Expert masks are evaluated only by ai4mars.paper_evaluate against a frozen checkpoint, never during training."
                    ],
                ),
                git_commit=current_git_commit(paths.project_root),
                random_seeds={"training": int(training["seed"])},
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
            training=TrainingRecord(
                optimizer=training["optimizer"],
                scheduler=training["scheduler"],
                learning_rate=float(training["learning_rate"]),
                class_weights=weights.tolist(),
                loss="CrossEntropyLoss",
                batch_size=physical_batch_size,
                physical_batch_size=physical_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                effective_batch_size=physical_batch_size * gradient_accumulation_steps,
                class_weighting_strategy=training["class_weighting"],
                epochs=int(training["epochs"]),
                precision_mode="amp" if training["mixed_precision"] else "float32",
            ),
            environment=EnvironmentRecord(
                python=platform.python_version(),
                pytorch=torch.__version__,
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            ),
        ),
    )

    logger.start()
    atomic_write_json(logger.run_dir / "config.json", snapshot)
    atomic_write_json(logger.run_dir / "artifacts" / "manifest_audit.json", audit)
    atomic_write_json(logger.run_dir / "artifacts" / "zero_valid_training_rows_audit.json", zero_valid_audit)

    start_epoch, global_step, best, stale = 1, 0, None, 0
    if training.get("resume_checkpoint"):
        state = load_training_checkpoint(
            model,
            optimizer,
            training["resume_checkpoint"],
            device,
            scheduler=scheduler,
            scaler=scaler,
            expected_metadata=metadata,
        )
        start_epoch, global_step, best = (
            state["epoch"] + 1,
            state["global_step"],
            state["best_validation_metric"],
        )

    try:
        for epoch in range(start_epoch, int(training["epochs"]) + 1):
            started = perf_counter()
            train_result = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                weighted_loss,
                device,
                epoch=epoch,
                amp_enabled=bool(training["mixed_precision"]),
                scaler=scaler,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            train_loss = train_result["mean_loss"]
            global_step += train_result["optimizer_steps"]

            if epoch % int(training["validation_interval"]):
                continue

            metrics = evaluate(
                model,
                loaders["val"],
                weighted_loss,
                device,
                return_detailed_metrics=True,
                unweighted_loss_fn=unweighted_loss,
            )
            improved = best is None or metrics["mean_iou"] > best
            best = max(best or 0.0, metrics["mean_iou"])
            stale = 0 if improved else stale + 1

            if scheduler is not None:
                if training["scheduler"] == "plateau":
                    scheduler.step(metrics["mean_iou"])
                else:
                    scheduler.step()

            last_checkpoint = logger.run_dir / "checkpoints" / "last.pth"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                last_checkpoint,
                metadata,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                best_validation_metric=best,
            )
            checkpoint = last_checkpoint
            if improved:
                best_checkpoint = logger.run_dir / "checkpoints" / "best_val_miou.pth"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_checkpoint,
                    metadata,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    best_validation_metric=best,
                )
                checkpoint = best_checkpoint

            per_class = {
                name: ClassMetrics(**{key: value for key, value in row.items() if key != "class_index"})
                for name, row in zip(CLASS_NAMES, metrics["per_class"])
            }
            logger.log_epoch(
                EpochMetrics(
                    timestamp=datetime.now(timezone.utc),
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=metrics["val_loss"],
                    pixel_accuracy=metrics["pixel_accuracy"],
                    mean_iou=metrics["mean_iou"],
                    per_class=per_class,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    epoch_duration_seconds=perf_counter() - started,
                    confusion_matrix=metrics["confusion_matrix"],
                    checkpoint=ArtifactRef(path=f"checkpoints/{checkpoint.name}", kind="checkpoint"),
                )
            )

            if training.get("early_stopping_patience") is not None and stale >= int(training["early_stopping_patience"]):
                break
    except BaseException as error:
        logger.fail(error)
        raise

    logger.finish(status=RunStatus.COMPLETED)
    return logger.run_dir


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
