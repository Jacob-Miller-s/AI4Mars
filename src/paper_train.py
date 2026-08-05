"""Paper-aligned DeepLabv3+ training implementation, invoked by ``python -m src.train``."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

from src.dataset import AI4MarsDataset, load_pairs_from_manifest
from src.foundation import build_checkpoint_metadata, current_git_commit, sha256_file
from src.paper_model import DeepLabV3PlusSpec, build_deeplabv3plus, paper_padding_metadata, validate_deeplabv3plus_spec
from src.paper_reproduction import CLASS_NAMES, assert_no_reproduction_leakage, compute_paper_class_composition, summarize_reproduction_manifests, validate_manifest_files
from src.research_console.run_store import RunLogger, atomic_write_json
from src.research_console.schema import ArtifactRef, ClassMetrics, EnvironmentRecord, EpochMetrics, ModelRecord, ProtocolRecord, ProvenanceRecord, RunMetadata, RunStatus, SplitRole, TrainingRecord
from src.runtime import resolve_runtime_paths
from src.train_utils import evaluate, load_training_checkpoint, save_checkpoint, train_one_epoch


WEIGHTING_STRATEGIES = {"paper_complement_composition", "inverse_frequency"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    for name in ("dataset-root", "manifest-root", "output-root", "run-id", "architecture", "backbone", "pretrained-weights", "optimizer", "scheduler", "class-weighting", "resume-checkpoint"):
        parser.add_argument(f"--{name}")
    for name in ("seed", "resolution", "batch-size", "gradient-accumulation", "epochs", "weight-decay", "learning-rate", "ignore-index", "num-workers", "checkpoint-interval", "validation-interval", "early-stopping", "batch-event-logging-interval"):
        parser.add_argument(f"--{name}", type=float if name in {"weight-decay", "learning-rate"} else int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-level", choices=("metadata", "full"), default="metadata")
    return parser.parse_args()


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {key!r} must be a mapping.")
    return value


def _override(section: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        section[key] = value


def load_and_validate_config(path: Path, args: argparse.Namespace | None = None) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    runtime, data, model, training, logging = (_section(config, key) for key in ("runtime", "data", "model", "training", "logging"))
    config.update(runtime=runtime, data=data, model=model, training=training, logging=logging)
    if args is not None:
        for key, value in (("dataset_root", args.dataset_root), ("manifest_root", args.manifest_root), ("output_root", args.output_root), ("run_id", args.run_id), ("accelerator", args.device)):
            _override(runtime, key, value)
        for key, value in (("seed", args.seed), ("batch_size", args.batch_size), ("gradient_accumulation_steps", args.gradient_accumulation), ("epochs", args.epochs), ("optimizer", args.optimizer), ("scheduler", args.scheduler), ("weight_decay", args.weight_decay), ("learning_rate", args.learning_rate), ("ignore_index", args.ignore_index), ("num_workers", args.num_workers), ("checkpoint_interval", args.checkpoint_interval), ("validation_interval", args.validation_interval), ("early_stopping_patience", args.early_stopping), ("batch_log_interval", args.batch_event_logging_interval), ("class_weighting", args.class_weighting), ("mixed_precision", args.amp), ("resume_checkpoint", args.resume_checkpoint)):
            _override(training, key, value)
        for key, value in (("architecture", args.architecture), ("backbone", args.backbone), ("pretrained_weights", args.pretrained_weights)):
            _override(model, key, value)
        if args.resolution is not None:
            model["input_size"] = [args.resolution, args.resolution]
    spec = DeepLabV3PlusSpec(architecture=model.get("architecture", ""), backbone=model.get("backbone", ""), pretrained_weights=model.get("pretrained_weights", ""), output_stride=int(model.get("output_stride", 16)), input_size=tuple(model.get("input_size", ())), num_classes=int(model.get("num_classes", 4)), ignore_index=int(training.get("ignore_index", 255)))
    validate_deeplabv3plus_spec(spec)
    required = {"dataset_manifest", "train_manifest", "val_manifest"}
    if missing := sorted(name for name in required if not data.get(name)):
        raise ValueError(f"Missing required reproduction manifests: {missing}")
    if training.get("class_weighting") not in WEIGHTING_STRATEGIES:
        raise ValueError(f"class_weighting must be one of {sorted(WEIGHTING_STRATEGIES)}.")
    if training.get("optimizer") not in {"adamw", "sgd"} or training.get("scheduler") not in {"none", "cosine", "plateau"}:
        raise ValueError("Unsupported optimizer or scheduler.")
    for name in ("batch_size", "gradient_accumulation_steps", "epochs", "checkpoint_interval", "validation_interval", "batch_log_interval"):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"training.{name} must be at least 1.")
    if training.get("max_samples_per_split") is not None and int(training["max_samples_per_split"]) < 1:
        raise ValueError("training.max_samples_per_split must be at least 1 when configured.")
    if float(training.get("learning_rate", 0)) <= 0 or float(training.get("weight_decay", 0)) < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative.")
    config["paper_model_spec"] = spec
    return config


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _weights(train_manifest: Path, strategy: str) -> tuple[torch.Tensor, dict[str, Any]]:
    composition = compute_paper_class_composition(train_manifest)
    if strategy == "paper_complement_composition":
        values = composition.class_weights
    else:
        inverse = [1.0 / max(value, 1e-8) for value in composition.class_proportions]
        mean = sum(inverse) / len(inverse)
        values = tuple(value / mean for value in inverse)
    return torch.tensor(values, dtype=torch.float32), {"strategy": strategy, "manifest_sha256": composition.manifest_sha256, "pixel_counts": list(composition.pixel_counts), "class_proportions": list(composition.class_proportions), "class_weights": list(values), "ignore_pixel_count": composition.ignore_pixel_count}


def _optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = {"lr": float(training["learning_rate"]), "weight_decay": float(training["weight_decay"])}
    return torch.optim.AdamW(model.parameters(), **kwargs) if training["optimizer"] == "adamw" else torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)


def _scheduler(optimizer: torch.optim.Optimizer, training: dict[str, Any]):
    if training["scheduler"] == "none": return None
    if training["scheduler"] == "cosine": return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")


def main() -> None:
    args = parse_args(); config = load_and_validate_config(Path(args.config), args)
    runtime, data, training, logging, spec = config["runtime"], config["data"], config["training"], config["logging"], config["paper_model_spec"]
    paths = resolve_runtime_paths(project_root=Path(__file__).parent.parent, dataset_root=runtime.get("dataset_root"), manifest_root=runtime.get("manifest_root"), output_root=runtime.get("output_root"), run_id=runtime.get("run_id"), accelerator=runtime.get("accelerator"))
    paths.ensure_writable_roots()
    device = torch.device("cuda" if paths.accelerator == "cuda" or (paths.accelerator == "auto" and torch.cuda.is_available()) else "cpu")
    if paths.accelerator == "cuda" and device.type != "cuda": raise RuntimeError("CUDA was requested but is unavailable.")
    if training["mixed_precision"] and device.type != "cuda": raise ValueError("mixed_precision requires CUDA.")
    manifests = {"train": paths.manifest_root / data["train_manifest"], "val": paths.manifest_root / data["val_manifest"]}
    expert_manifest_keys = {"expert_min1_100agree": "expert_min1_manifest", "expert_min2_100agree": "expert_min2_manifest", "expert_min3_100agree": "expert_min3_manifest"}
    expert_manifests = {name: paths.manifest_root / data[key] for name, key in expert_manifest_keys.items() if data.get(key)}
    audit_manifests = {**manifests, **expert_manifests}
    assert_no_reproduction_leakage(audit_manifests)
    if args.validate_only and args.validation_level == "full":
        for name, path in audit_manifests.items(): validate_manifest_files(path, paths.dataset_root, split_name=name)
    audit = summarize_reproduction_manifests(audit_manifests)
    if args.validate_only:
        print(json.dumps(audit, indent=2, sort_keys=True)); return
    _seed(int(training["seed"]))
    pairs = {name: load_pairs_from_manifest(path, dataset_root=paths.dataset_root, required_label_scheme="NAV", require_shape_match=True) for name, path in manifests.items()}
    if training.get("max_samples_per_split") is not None:
        sample_limit = int(training["max_samples_per_split"])
        pairs = {name: values[:sample_limit] for name, values in pairs.items()}
    options = {"image_size": spec.input_size, "require_original_shape_match": True, "normalization_mean": spec.normalization_mean, "normalization_std": spec.normalization_std}
    datasets = {name: AI4MarsDataset(value, **options) for name, value in pairs.items()}
    loader_options = {"batch_size": int(training["batch_size"]), "num_workers": int(training["num_workers"]), "pin_memory": device.type == "cuda", "persistent_workers": int(training["num_workers"]) > 0}
    loaders = {name: DataLoader(dataset, shuffle=name == "train", **loader_options) for name, dataset in datasets.items()}
    model = build_deeplabv3plus(spec).to(device); weights, weighting = _weights(manifests["train"], training["class_weighting"]); weights = weights.to(device)
    weighted_loss, unweighted_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=spec.ignore_index), nn.CrossEntropyLoss(ignore_index=spec.ignore_index)
    optimizer, scheduler, scaler = _optimizer(model, training), None, torch.amp.GradScaler("cuda", enabled=bool(training["mixed_precision"]))
    scheduler = _scheduler(optimizer, training)
    physical_batch_size, gradient_accumulation_steps = int(training["batch_size"]), int(training["gradient_accumulation_steps"])
    padding_metadata = paper_padding_metadata(spec)
    metadata = build_checkpoint_metadata(project_root=paths.project_root, dataset_manifest_path=paths.manifest_root / data["dataset_manifest"], split_manifest_paths=audit_manifests, active_split_name="val", preprocessing={"input_size": list(spec.input_size), "normalization_mean": list(spec.normalization_mean), "normalization_std": list(spec.normalization_std), "mask_interpolation": "nearest", **padding_metadata}, loss_name="CrossEntropyLoss", loss_weights=weights.tolist(), model_name=f"{spec.architecture}/{spec.backbone}", seed=int(training["seed"]))
    snapshot = config | {"paper_model_spec": spec.metadata()}; metadata.update({"paper_reproduction": True, "model": spec.metadata(), "class_weighting": weighting, "configuration": snapshot})
    logger = RunLogger(paths.event_root, RunMetadata(run_id=paths.run_id, experiment_name=config.get("experiment_name", paths.run_id), paper_reproduction=True, tags=["paper-reproduction", "msl", "navcam", "deeplabv3plus"], provenance=ProvenanceRecord(dataset_name="AI4Mars", dataset_version="ai4mars-dataset-merged-0.6", dataset_manifest_sha256=sha256_file(paths.manifest_root / data["dataset_manifest"]), split_manifest_hashes={name: sha256_file(path) for name, path in audit_manifests.items()}, split_role=SplitRole.CROWDSOURCED_VALIDATION, protocol=ProtocolRecord(valid=True, notes=["Expert masks are evaluated only by src.paper_evaluate against a frozen checkpoint, never during training."]), git_commit=current_git_commit(paths.project_root), random_seeds={"training": int(training["seed"])}), model=ModelRecord(name=spec.architecture, encoder=spec.backbone, pretrained_weights=spec.pretrained_weights, parameter_count=sum(item.numel() for item in model.parameters()), input_resolution=spec.input_size, requested_input_size=tuple(padding_metadata["requested_input_size"]), internal_padding_multiple=padding_metadata["internal_padding_multiple"], internal_padded_size_for_513=tuple(padding_metadata["internal_padded_size_for_513"]), input_padding_policy=padding_metadata["input_padding_policy"], input_padding_mode=padding_metadata["input_padding_mode"], normalized_padding_value=padding_metadata["normalized_padding_value"], output_crop_policy=padding_metadata["output_crop_policy"]), training=TrainingRecord(optimizer=training["optimizer"], scheduler=training["scheduler"], learning_rate=float(training["learning_rate"]), class_weights=weights.tolist(), loss="CrossEntropyLoss", batch_size=physical_batch_size, physical_batch_size=physical_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, effective_batch_size=physical_batch_size * gradient_accumulation_steps, class_weighting_strategy=training["class_weighting"], epochs=int(training["epochs"]), precision_mode="amp" if training["mixed_precision"] else "float32"), environment=EnvironmentRecord(python=platform.python_version(), pytorch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)))
    logger.start(); atomic_write_json(logger.run_dir / "config.json", snapshot); atomic_write_json(logger.run_dir / "artifacts" / "manifest_audit.json", audit)
    start_epoch, global_step, best, stale = 1, 0, None, 0
    if training.get("resume_checkpoint"):
        state = load_training_checkpoint(model, optimizer, training["resume_checkpoint"], device, scheduler=scheduler, scaler=scaler, expected_metadata=metadata); start_epoch, global_step, best = state["epoch"] + 1, state["global_step"], state["best_validation_metric"]
    try:
        for epoch in range(start_epoch, int(training["epochs"]) + 1):
            started = perf_counter(); train_result = train_one_epoch(model, loaders["train"], optimizer, weighted_loss, device, epoch=epoch, run_logger=logger, batch_log_interval=int(training["batch_log_interval"]), amp_enabled=bool(training["mixed_precision"]), scaler=scaler, gradient_accumulation_steps=gradient_accumulation_steps); train_loss = train_result["mean_loss"]; global_step += train_result["optimizer_steps"]
            if epoch % int(training["validation_interval"]): continue
            metrics = evaluate(model, loaders["val"], weighted_loss, device, return_detailed_metrics=True, unweighted_loss_fn=unweighted_loss); improved = best is None or metrics["mean_iou"] > best; best = max(best or 0.0, metrics["mean_iou"]); stale = 0 if improved else stale + 1
            if scheduler is not None: scheduler.step(metrics["mean_iou"]) if training["scheduler"] == "plateau" else scheduler.step()
            last_checkpoint = logger.run_dir / "checkpoints" / "last.pth"
            save_checkpoint(model, optimizer, epoch, last_checkpoint, metadata, scheduler=scheduler, scaler=scaler, global_step=global_step, best_validation_metric=best)
            checkpoint = last_checkpoint
            if improved:
                best_checkpoint = logger.run_dir / "checkpoints" / "best_val_miou.pth"
                save_checkpoint(model, optimizer, epoch, best_checkpoint, metadata, scheduler=scheduler, scaler=scaler, global_step=global_step, best_validation_metric=best)
                checkpoint = best_checkpoint
            per_class = {name: ClassMetrics(**{key: value for key, value in row.items() if key != "class_index"}) for name, row in zip(CLASS_NAMES, metrics["per_class"])}
            logger.log_epoch(EpochMetrics(timestamp=datetime.now(timezone.utc), epoch=epoch, train_loss=train_loss, val_loss=metrics["val_loss"], pixel_accuracy=metrics["pixel_accuracy"], mean_iou=metrics["mean_iou"], per_class=per_class, learning_rate=optimizer.param_groups[0]["lr"], epoch_duration_seconds=perf_counter() - started, confusion_matrix=metrics["confusion_matrix"], checkpoint=ArtifactRef(path=f"checkpoints/{checkpoint.name}", kind="checkpoint")))
            if training.get("early_stopping_patience") is not None and stale >= int(training["early_stopping_patience"]): break
    except BaseException as error:
        logger.fail(error); raise
    logger.finish(status=RunStatus.COMPLETED)


if __name__ == "__main__": main()