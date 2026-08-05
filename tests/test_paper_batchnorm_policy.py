import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src import paper_train
from src.paper_train import _build_dataloaders
from src.train_utils import evaluate, train_one_epoch


class TinyAsppLikeBatchNormNet(nn.Module):
    """Minimal network that reproduces ASPP-style pooled BatchNorm behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_bn = nn.BatchNorm2d(8)
        self.head = nn.Conv2d(8, 4, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        features = torch.relu(self.stem(inputs))
        pooled = self.pool_bn(self.pool(features))
        pooled = F.interpolate(pooled, size=(height, width), mode="nearest")
        return self.head(features + pooled)


class BatchNormPolicyValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_config(self, *, batch_size: int, grad_accum: int = 1) -> Path:
        config = self.root / "config.yaml"
        config.write_text(
            "runtime: {}\n"
            "data:\n"
            "  dataset_manifest: dataset_manifest.csv\n"
            "  train_manifest: train.csv\n"
            "  val_manifest: val.csv\n"
            "  expert_min1_manifest: expert_min1.csv\n"
            "  expert_min2_manifest: expert_min2.csv\n"
            "  expert_min3_manifest: expert_min3.csv\n"
            "model:\n"
            "  architecture: DeepLabV3Plus\n"
            "  backbone: resnet101\n"
            "  pretrained_weights: imagenet\n"
            "  output_stride: 16\n"
            "  input_size: [513, 513]\n"
            "  num_classes: 4\n"
            "training:\n"
            "  seed: 42\n"
            f"  batch_size: {batch_size}\n"
            f"  gradient_accumulation_steps: {grad_accum}\n"
            "  epochs: 1\n"
            "  optimizer: adamw\n"
            "  learning_rate: 0.0001\n"
            "  scheduler: none\n"
            "  weight_decay: 0.0\n"
            "  class_weighting: paper_complement_composition\n"
            "  ignore_index: 255\n"
            "  num_workers: 0\n"
            "  mixed_precision: false\n"
            "  checkpoint_interval: 1\n"
            "  validation_interval: 1\n"
            "  early_stopping_patience: null\n"
            "  batch_log_interval: 1\n"
            "logging: {}\n",
            encoding="utf-8",
        )
        return config

    def test_rejects_training_batch_size_one(self) -> None:
        config = self._write_config(batch_size=1, grad_accum=4)
        with self.assertRaisesRegex(ValueError, "training.batch_size must be at least 2"):
            paper_train.load_and_validate_config(config)

    def test_accepts_training_batch_size_two(self) -> None:
        config = self._write_config(batch_size=2, grad_accum=4)
        loaded = paper_train.load_and_validate_config(config)
        self.assertEqual(int(loaded["training"]["batch_size"]), 2)
        self.assertEqual(int(loaded["training"]["gradient_accumulation_steps"]), 4)

    def test_eval_path_can_allow_batch_size_one(self) -> None:
        config = self._write_config(batch_size=1, grad_accum=4)
        loaded = paper_train.load_and_validate_config(config, enforce_training_batch_size=False)
        self.assertEqual(int(loaded["training"]["batch_size"]), 1)


class DataLoaderBatchPolicyTests(unittest.TestCase):
    def test_split_specific_drop_last_policy(self) -> None:
        images = torch.zeros((5, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((5, 8, 8), dtype=torch.long)
        datasets = {
            "train": TensorDataset(images, masks),
            "val": TensorDataset(images[:2], masks[:2]),
            "expert_min1": TensorDataset(images[:1], masks[:1]),
        }
        loaders = _build_dataloaders(datasets, batch_size=2, num_workers=0, pin_memory=False)

        self.assertTrue(loaders["train"].drop_last)
        self.assertFalse(loaders["val"].drop_last)
        self.assertFalse(loaders["expert_min1"].drop_last)

    def test_five_sample_training_set_drops_final_singleton(self) -> None:
        images = torch.zeros((5, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((5, 8, 8), dtype=torch.long)
        loaders = _build_dataloaders(
            {"train": TensorDataset(images, masks)},
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        batches = list(loaders["train"])
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0].shape[0], 2)
        self.assertEqual(batches[1][0].shape[0], 2)


class BatchNormRuntimeBehaviorTests(unittest.TestCase):
    def test_train_mode_batchnorm_fails_for_singleton_batch(self) -> None:
        model = TinyAsppLikeBatchNormNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((1, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((1, 8, 8), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False, drop_last=False)

        with self.assertRaisesRegex(ValueError, "Expected more than 1 value per channel"):
            train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

    def test_eval_mode_validation_succeeds_with_batch_size_one(self) -> None:
        model = TinyAsppLikeBatchNormNet()
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((2, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((2, 8, 8), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False, drop_last=False)

        metrics = evaluate(model, loader, loss_fn, torch.device("cpu"))

        self.assertIn("mean_iou", metrics)
        self.assertIn("pixel_accuracy", metrics)

    def test_end_to_end_train_uses_drop_last_to_avoid_singleton_batch(self) -> None:
        model = TinyAsppLikeBatchNormNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((5, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((5, 8, 8), dtype=torch.long)
        loaders = _build_dataloaders(
            {"train": TensorDataset(images, masks)},
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )

        result = train_one_epoch(model, loaders["train"], optimizer, loss_fn, torch.device("cpu"))

        self.assertEqual(result["optimizer_steps"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["mean_loss"])))


class BatchSizeMetadataRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset_root = self.root / "ai4mars-dataset-merged-0.6"
        self.manifest_root = self.root / "manifests"
        self.output_root = self.root / "outputs"
        self.dataset_root.mkdir()
        self.manifest_root.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _row(self, source_id: str, sequence_id: str, *, agreement: str = "") -> dict[str, str]:
        return {
            "dataset_relative_image_path": f"msl/ncam/images/{source_id}.JPG",
            "dataset_relative_mask_path": f"msl/ncam/labels/{source_id}.png",
            "stable_source_image_id": source_id,
            "sequence_id": sequence_id,
            "mission": "msl",
            "rover": "curiosity",
            "camera": "ncam",
            "label_scheme": "NAV",
            "label_role": "expert_gold_test" if agreement else "crowdsourced_train",
            "agreement_threshold": agreement,
            "image_width": "8",
            "image_height": "8",
            "mask_width": "8",
            "mask_height": "8",
            "per_class_pixel_counts_json": json.dumps({"0": 40, "1": 10, "2": 10, "3": 4}),
        }

    def _write_manifest(self, name: str, rows: list[dict[str, str]]) -> None:
        path = self.manifest_root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_real_files(self, source_ids: list[str]) -> None:
        images_dir = self.dataset_root / "msl" / "ncam" / "images"
        labels_dir = self.dataset_root / "msl" / "ncam" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for source_id in source_ids:
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images_dir / f"{source_id}.JPG")
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(labels_dir / f"{source_id}.png")

    def _write_config(self) -> Path:
        config = self.root / "config.yaml"
        config.write_text(
            "runtime:\n"
            "  run_id: metadata-batch-size-test\n"
            "data:\n"
            "  dataset_manifest: dataset_manifest.csv\n"
            "  train_manifest: train.csv\n"
            "  val_manifest: val.csv\n"
            "  expert_min1_manifest: expert_min1.csv\n"
            "  expert_min2_manifest: expert_min2.csv\n"
            "  expert_min3_manifest: expert_min3.csv\n"
            "model:\n"
            "  architecture: DeepLabV3Plus\n"
            "  backbone: resnet101\n"
            "  pretrained_weights: imagenet\n"
            "  output_stride: 16\n"
            "  input_size: [513, 513]\n"
            "  num_classes: 4\n"
            "training:\n"
            "  seed: 42\n"
            "  batch_size: 2\n"
            "  gradient_accumulation_steps: 3\n"
            "  epochs: 1\n"
            "  optimizer: adamw\n"
            "  learning_rate: 0.0001\n"
            "  scheduler: none\n"
            "  weight_decay: 0.0\n"
            "  class_weighting: paper_complement_composition\n"
            "  ignore_index: 255\n"
            "  num_workers: 0\n"
            "  mixed_precision: false\n"
            "  checkpoint_interval: 1\n"
            "  validation_interval: 1\n"
            "  early_stopping_patience: null\n"
            "  batch_log_interval: 1\n"
            "logging: {}\n",
            encoding="utf-8",
        )
        return config

    def test_recorded_physical_and_effective_batch_sizes(self) -> None:
        self._write_real_files(["A", "B", "C", "D", "E", "X1", "X2", "X3"])
        self._write_manifest("train.csv", [self._row("A", "SEQ_A"), self._row("B", "SEQ_B"), self._row("C", "SEQ_C"), self._row("D", "SEQ_D")])
        self._write_manifest("val.csv", [self._row("E", "SEQ_E")])
        self._write_manifest("expert_min1.csv", [self._row("X1", "SEQ_X1", agreement="min1-100agree")])
        self._write_manifest("expert_min2.csv", [self._row("X2", "SEQ_X2", agreement="min2-100agree")])
        self._write_manifest("expert_min3.csv", [self._row("X3", "SEQ_X3", agreement="min3-100agree")])
        (self.manifest_root / "dataset_manifest.csv").write_text("placeholder\n", encoding="utf-8")
        config_path = self._write_config()

        def fake_train_one_epoch(*args, **kwargs):
            return {"mean_loss": 0.1, "optimizer_steps": 2}

        def fake_evaluate(*args, **kwargs):
            return {
                "val_loss": 1.0,
                "pixel_accuracy": 0.5,
                "mean_iou": 0.25,
                "per_class": [
                    {"class_index": i, "support": 1, "predicted": 1, "true_positive": 1, "false_positive": 0, "false_negative": 0, "iou": 1.0, "dice_f1": 1.0, "precision": 1.0, "recall": 1.0}
                    for i in range(4)
                ],
                "confusion_matrix": [[1, 0, 0, 0] for _ in range(4)],
            }

        argv = [
            "paper_train.py",
            "--config", str(config_path),
            "--dataset-root", str(self.dataset_root),
            "--manifest-root", str(self.manifest_root),
            "--output-root", str(self.output_root),
        ]

        with patch.object(sys, "argv", argv), \
            patch.object(paper_train, "build_deeplabv3plus", lambda spec: nn.Conv2d(3, 4, kernel_size=1)), \
            patch.object(paper_train, "train_one_epoch", fake_train_one_epoch), \
            patch.object(paper_train, "evaluate", fake_evaluate), \
            contextlib.redirect_stdout(io.StringIO()):
            paper_train.main()

        metadata_path = self.output_root / "runs" / "metadata-batch-size-test" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training = metadata["training"]
        self.assertEqual(training["physical_batch_size"], 2)
        self.assertEqual(training["gradient_accumulation_steps"], 3)
        self.assertEqual(training["effective_batch_size"], 6)


if __name__ == "__main__":
    unittest.main()
