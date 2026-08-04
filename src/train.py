"""Non-interactive reproducible AI4Mars segmentation training entry point."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset import AI4MarsDataset, load_pairs_from_manifest
from src.foundation import build_checkpoint_metadata, current_git_commit, sha256_file
from src.research_console.run_store import RunLogger
from src.research_console.schema import ArtifactRef, ClassMetrics, EnvironmentRecord, EpochMetrics, ModelRecord, ProtocolRecord, ProvenanceRecord, RunMetadata, RunStatus, SplitRole, TrainingRecord
from src.runtime import resolve_runtime_paths
from src.train_utils import load_training_checkpoint, save_checkpoint, train_one_epoch, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root"); parser.add_argument("--manifest-root"); parser.add_argument("--output-root")
    parser.add_argument("--checkpoint-root"); parser.add_argument("--event-root"); parser.add_argument("--cache-root")
    parser.add_argument("--run-id"); parser.add_argument("--accelerator", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--resume"); parser.add_argument("--epochs", type=int); parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int); parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int); parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runtime_config, data, model_config, training, logging = (config.get(key, {}) for key in ("runtime", "data", "model", "training", "logging"))
    paths = resolve_runtime_paths(project_root=Path(__file__).parent.parent, dataset_root=args.dataset_root or runtime_config.get("dataset_root"), manifest_root=args.manifest_root or runtime_config.get("manifest_root"), output_root=args.output_root or runtime_config.get("output_root"), checkpoint_root=args.checkpoint_root or runtime_config.get("checkpoint_root"), event_root=args.event_root or runtime_config.get("event_root"), cache_root=args.cache_root or runtime_config.get("cache_root"), run_id=args.run_id or runtime_config.get("run_id"), accelerator=args.accelerator or runtime_config.get("accelerator"))
    paths.ensure_writable_roots()
    if paths.accelerator == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if paths.accelerator == "cuda" or (paths.accelerator == "auto" and torch.cuda.is_available()) else "cpu")
    amp = training.get("mixed_precision", False) if args.amp is None else args.amp
    if amp and device.type != "cuda": raise ValueError("mixed_precision requires a CUDA device.")
    seed = int(training.get("seed", 42)); seed_everything(seed)
    image_size = int(args.image_size or training.get("image_size", 256)); batch_size = int(args.batch_size or training.get("batch_size", 4))
    train_manifest = paths.manifest_root / data.get("train_manifest", "splits/msl_navcam_v1/train_nav.csv")
    val_manifest = paths.manifest_root / data.get("val_manifest", "splits/msl_navcam_v1/val_nav.csv")
    dataset_manifest = paths.manifest_root / data.get("dataset_manifest", "ai4mars_dataset_manifest.csv")
    train_pairs = load_pairs_from_manifest(train_manifest, dataset_root=paths.dataset_root, required_label_scheme=data.get("label_scheme", "NAV"), require_shape_match=True)
    val_pairs = load_pairs_from_manifest(val_manifest, dataset_root=paths.dataset_root, required_label_scheme=data.get("label_scheme", "NAV"), require_shape_match=True)
    loader_kwargs = {"batch_size": batch_size, "num_workers": int(args.num_workers if args.num_workers is not None else training.get("num_workers", 0)), "pin_memory": device.type == "cuda" and training.get("pin_memory", True)}
    train_loader = DataLoader(AI4MarsDataset(train_pairs, (image_size, image_size), require_original_shape_match=True), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(AI4MarsDataset(val_pairs, (image_size, image_size), require_original_shape_match=True), shuffle=False, **loader_kwargs)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=model_config.get("encoder", "resnet34"), encoder_weights=model_config.get("pretrained_encoder", "imagenet"), in_channels=3, classes=4).to(device)
    weights = torch.tensor(training["class_weights"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate or training.get("learning_rate", 1e-3)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    metadata = build_checkpoint_metadata(project_root=paths.project_root, dataset_manifest_path=dataset_manifest, split_manifest_paths={"train": train_manifest, "val": val_manifest}, active_split_name="val", preprocessing={"image_size": [image_size, image_size], "require_original_shape_match": True, "label_scheme": data.get("label_scheme", "NAV")}, loss_name="CrossEntropyLoss", loss_weights=weights.tolist(), model_name=f"Unet/{model_config.get('encoder', 'resnet34')}", seed=seed)
    logger = RunLogger(paths.event_root, RunMetadata(run_id=paths.run_id, experiment_name=config.get("experiment_name", paths.run_id), provenance=ProvenanceRecord(dataset_name="AI4Mars", dataset_version="ai4mars-dataset-merged-0.6", dataset_manifest_sha256=sha256_file(dataset_manifest), split_manifest_hashes={"train": sha256_file(train_manifest), "val": sha256_file(val_manifest)}, split_role=SplitRole.CROWDSOURCED_VALIDATION, protocol=ProtocolRecord(valid=True), git_commit=current_git_commit(paths.project_root), random_seeds={"training": seed}), model=ModelRecord(name="Unet", encoder=model_config.get("encoder", "resnet34"), pretrained_weights=model_config.get("pretrained_encoder", "imagenet"), input_resolution=(image_size, image_size)), training=TrainingRecord(optimizer="Adam", learning_rate=optimizer.param_groups[0]["lr"], class_weights=weights.tolist(), loss="CrossEntropyLoss", batch_size=batch_size, epochs=int(args.epochs or training.get("epochs", 20)), precision_mode="amp" if amp else "float32"), environment=EnvironmentRecord(pytorch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)))
    logger.start(); start_epoch = 1; global_step = 0; best = None
    if args.resume:
        state = load_training_checkpoint(model, optimizer, args.resume, device, scaler=scaler, expected_metadata=metadata); start_epoch, global_step, best = state["epoch"] + 1, state["global_step"], state["best_validation_metric"]
    try:
        for epoch in range(start_epoch, int(args.epochs or training.get("epochs", 20)) + 1):
            started = perf_counter(); loss = train_one_epoch(model, train_loader, optimizer, nn.CrossEntropyLoss(weight=weights, ignore_index=255), device, epoch=epoch, run_logger=logger, batch_log_interval=int(logging.get("batch_log_interval", 100)), amp_enabled=amp, scaler=scaler); global_step += len(train_loader)
            metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(weight=weights, ignore_index=255), device, return_detailed_metrics=True, return_per_class_iou=True)
            improved = best is None or metrics["mean_iou"] > best; best = max(best or 0.0, metrics["mean_iou"])
            checkpoint = logger.run_dir / "checkpoints" / ("best.pth" if improved else f"epoch_{epoch:03d}.pth")
            if improved or epoch % int(training.get("checkpoint_interval", 1)) == 0: save_checkpoint(model, optimizer, epoch, checkpoint, metadata, scaler=scaler, global_step=global_step, best_validation_metric=best)
            per_class = {name: ClassMetrics(**{key: value for key, value in item.items() if key != "class_index"}) for name, item in zip(("soil", "bedrock", "sand", "big_rock"), metrics["per_class"])}
            logger.log_epoch(EpochMetrics(timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc), epoch=epoch, train_loss=loss, val_loss=metrics["val_loss"], pixel_accuracy=metrics["pixel_acc"], mean_iou=metrics["mean_iou"], per_class=per_class, learning_rate=optimizer.param_groups[0]["lr"], epoch_duration_seconds=perf_counter()-started, confusion_matrix=metrics["confusion_matrix"], checkpoint=ArtifactRef(path=f"checkpoints/{checkpoint.name}", kind="checkpoint")))
    except BaseException as error: logger.fail(error); raise
    logger.finish(status=RunStatus.COMPLETED)

if __name__ == "__main__": main()