"""Clearly labeled CPU-only synthetic smoke training for the research console."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.train_utils import evaluate, save_checkpoint, train_one_epoch

from .run_store import RunLogger, append_jsonl
from .schema import (
    ArtifactRef,
    EnvironmentRecord,
    ModelRecord,
    ProtocolRecord,
    ProvenanceRecord,
    RunMetadata,
    RunStatus,
    SplitRole,
    TrainingRecord,
)


CLASS_COLORS = np.array(
    [
        [183, 104, 69],
        [60, 124, 104],
        [200, 155, 56],
        [121, 82, 72],
    ],
    dtype=np.uint8,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _DelayedLogger:
    """Delay only the synthetic demo's emitted batches so live updates are observable."""

    def __init__(self, logger: RunLogger, delay_seconds: float) -> None:
        self._logger = logger
        self._delay_seconds = delay_seconds

    def log_batch(self, **kwargs: Any) -> None:
        self._logger.log_batch(**kwargs)
        if self._delay_seconds:
            time.sleep(self._delay_seconds)

    def log_epoch(self, event: Any) -> None:
        self._logger.log_epoch(event)


def build_demo_metadata(run_id: str, *, manifest_variant: str = "a") -> RunMetadata:
    """Create an intentionally invalid-for-benchmark record for synthetic data."""
    return RunMetadata(
        run_id=run_id,
        experiment_name="Synthetic console smoke training",
        hypothesis="Verify live observability and artifact plumbing with synthetic tensors only.",
        tags=["demo", "synthetic", "non-benchmark"],
        researcher_notes="Synthetic demo data. Never interpret these metrics as AI4Mars results.",
        provenance=ProvenanceRecord(
            dataset_name="Synthetic AI4Mars-shaped fixture",
            dataset_version="demo-v1",
            source_record="synthetic-fixture",
            dataset_manifest_sha256=_hash(f"synthetic-manifest-{manifest_variant}"),
            split_manifest_hashes={
                "train": _hash(f"synthetic-train-{manifest_variant}"),
                "val": _hash(f"synthetic-val-{manifest_variant}"),
            },
            split_role=SplitRole.CROWDSOURCED_VALIDATION,
            protocol=ProtocolRecord(
                valid=False,
                failed_gates=["synthetic_demo_not_research_benchmark"],
                notes=["Generated only for local console verification."],
            ),
            git_branch="synthetic-demo",
            git_dirty=False,
            random_seeds={"torch": 7, "numpy": 7},
            determinism={"torch_manual_seed": True, "cuda_used": False},
        ),
        model=ModelRecord(
            name="TinyConvSeg",
            encoder="none",
            pretrained_weights="none",
            input_resolution=(32, 32),
        ),
        training=TrainingRecord(
            optimizer="SGD",
            scheduler=None,
            learning_rate=0.1,
            loss="CrossEntropyLoss(ignore_index=255)",
            batch_size=2,
            epochs=3,
            augmentation={"synthetic": True},
            precision_mode="float32-cpu",
        ),
        environment=EnvironmentRecord(
            python=sys.version,
            pytorch=torch.__version__,
            gpu="not used by synthetic demo",
        ),
    )


def _synthetic_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    images = torch.rand((12, 3, 32, 32), generator=generator)
    masks = torch.randint(0, 4, (12, 32, 32), generator=generator, dtype=torch.long)
    masks[0, :2, :2] = 255
    return DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _write_demo_prediction_assets(run_dir: Path) -> list[ArtifactRef]:
    """Create small, labeled synthetic panels for workbench UI testing only."""
    samples_dir = run_dir / "artifacts" / "samples"
    index_path = run_dir / "artifacts" / "prediction_index.jsonl"
    refs = [ArtifactRef(path="artifacts/prediction_index.jsonl", kind="prediction_index")]
    rng = np.random.default_rng(9)
    for sample_index in range(4):
        original = rng.integers(45, 185, size=(96, 128, 3), dtype=np.uint8)
        truth = rng.integers(0, 4, size=(96, 128), dtype=np.uint8)
        prediction = truth.copy()
        prediction[20 + sample_index : 44 + sample_index, 30:72] = (sample_index + 1) % 4
        error = prediction != truth
        overlay = ((0.55 * original) + (0.45 * CLASS_COLORS[prediction])).astype(np.uint8)
        heatmap = np.zeros_like(original)
        heatmap[error] = [218, 81, 55]
        heatmap[~error] = [56, 112, 90]
        base = f"sample_{sample_index:02d}"
        assets = {
            "image": original,
            "ground_truth": CLASS_COLORS[truth],
            "prediction": CLASS_COLORS[prediction],
            "overlay": overlay,
            "error_heatmap": heatmap,
        }
        asset_paths = {}
        for name, array in assets.items():
            relative_path = f"artifacts/samples/{base}_{name}.png"
            _save_png(run_dir / Path(*relative_path.split("/")), array)
            asset_paths[name] = relative_path
            refs.append(ArtifactRef(path=relative_path, kind=f"demo_{name}"))
        image_iou = float((prediction == truth).sum() / prediction.size)
        append_jsonl(
            index_path,
            {
                "sample_id": base,
                "split": "synthetic_validation",
                "synthetic_demo": True,
                "image_iou": image_iou,
                "loss": float(1 - image_iou),
                "uncertainty": float(error.mean()),
                "big_rock_false_negative": sample_index % 2 == 0,
                "big_rock_to_soil": sample_index == 2,
                "assets": asset_paths,
            },
        )
    return refs


def run_smoke_training(
    runs_root: Path,
    *,
    run_id: str = "synthetic-smoke-run",
    epochs: int = 3,
    batch_delay_seconds: float = 0.0,
    manifest_variant: str = "a",
    fail_after_epoch: int | None = None,
) -> RunLogger:
    """Run a small CPU training job through the production logger interfaces."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    torch.manual_seed(7)
    metadata = build_demo_metadata(run_id, manifest_variant=manifest_variant)
    metadata = metadata.model_copy(update={"training": metadata.training.model_copy(update={"epochs": epochs})})
    logger = RunLogger(runs_root, metadata, system_sample_interval_seconds=0.1)
    logger.start()
    delayed_logger = _DelayedLogger(logger, batch_delay_seconds)
    device = torch.device("cpu")
    model = nn.Conv2d(3, 4, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    train_loader = _synthetic_loader()
    val_loader = _synthetic_loader()
    best_iou = float("-inf")
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                epoch=epoch,
                run_logger=delayed_logger,
                batch_log_interval=1,
            )
            results = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                return_detailed_metrics=True,
                epoch=epoch,
                train_loss=train_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
                epoch_duration_seconds=time.perf_counter() - epoch_started,
                run_logger=delayed_logger,
            )
            if results["mean_iou"] > best_iou:
                best_iou = results["mean_iou"]
                checkpoint_path = logger.run_dir / "checkpoints" / "best_synthetic.pth"
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    checkpoint_path,
                    metadata={"synthetic_demo": True, "run_id": run_id},
                )
                logger.register_artifact(
                    ArtifactRef(
                        path="checkpoints/best_synthetic.pth",
                        kind="checkpoint",
                        description="Synthetic demo checkpoint; not a research artifact.",
                    )
                )
            if fail_after_epoch == epoch:
                raise RuntimeError("Intentional synthetic smoke-training failure.")
        for artifact in _write_demo_prediction_assets(logger.run_dir):
            logger.register_artifact(artifact)
        logger.finish()
    except BaseException as error:
        logger.fail(error)
        raise
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic, non-benchmark console smoke training job.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs") / "runs")
    parser.add_argument("--run-id", default=f"synthetic-demo-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-delay-seconds", type=float, default=0.25)
    parser.add_argument("--manifest-variant", default="a")
    parser.add_argument("--fail-after-epoch", type=int)
    args = parser.parse_args()
    logger = run_smoke_training(
        args.runs_root,
        run_id=args.run_id,
        epochs=args.epochs,
        batch_delay_seconds=args.batch_delay_seconds,
        manifest_variant=args.manifest_variant,
        fail_after_epoch=args.fail_after_epoch,
    )
    print(logger.run_dir)


if __name__ == "__main__":
    main()