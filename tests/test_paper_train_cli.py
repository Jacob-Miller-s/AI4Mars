"""End-to-end tests for src/paper_train.py's --validate-only workflow.

These tests exercise the real ``main()`` entry point through the validate-only
path only, which returns before any model is constructed (see
``src/paper_train.py``: the ``if args.validate_only: ... return`` check runs
before ``build_deeplabv3plus``). This keeps the tests CPU-fast while still
proving the --validation-level flag and the audit-only handling of expert
manifests actually work end-to-end, not just in isolated unit calls.
"""

import contextlib
import csv
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src import paper_train


class _ManifestFixtureMixin:
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest_root = self.root / "manifests"
        self.manifest_root.mkdir()
        self.dataset_root = self.root / "ai4mars-dataset-merged-0.6"
        self.dataset_root.mkdir()
        self.output_root = self.root / "outputs"

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

    def _manifest(self, name: str, row: dict[str, str]) -> Path:
        path = self.manifest_root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def _write_real_files(self, source_id: str) -> None:
        images_dir = self.dataset_root / "msl" / "ncam" / "images"
        labels_dir = self.dataset_root / "msl" / "ncam" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images_dir / f"{source_id}.JPG")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0], mask[0, 1], mask[0, 2] = 1, 2, 3
        Image.fromarray(mask).save(labels_dir / f"{source_id}.png")

    def _write_config(self) -> Path:
        config = self.root / "config.yaml"
        config.write_text(
            "runtime: {}\n"
            "data:\n"
            "  dataset_manifest: unused.csv\n"
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
            "  batch_size: 1\n"
            "  gradient_accumulation_steps: 1\n"
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

    def _run_main(self, *, validation_level: str) -> dict:
        self._manifest("train.csv", self._row("A", "SEQ_A"))
        self._manifest("val.csv", self._row("B", "SEQ_B"))
        self._manifest("expert_min1.csv", self._row("C1", "SEQ_C1", agreement="min1-100agree"))
        self._manifest("expert_min2.csv", self._row("C2", "SEQ_C2", agreement="min2-100agree"))
        self._manifest("expert_min3.csv", self._row("C3", "SEQ_C3", agreement="min3-100agree"))
        config_path = self._write_config()
        argv = [
            "paper_train.py",
            "--config", str(config_path),
            "--dataset-root", str(self.dataset_root),
            "--manifest-root", str(self.manifest_root),
            "--output-root", str(self.output_root),
            "--validate-only",
            "--validation-level", validation_level,
        ]
        buffer = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(buffer):
            paper_train.main()
        return json.loads(buffer.getvalue())


class ValidateOnlyWorkflowTests(_ManifestFixtureMixin, unittest.TestCase):
    def test_metadata_level_audits_all_splits_without_dataset_files(self) -> None:
        audit = self._run_main(validation_level="metadata")

        self.assertEqual(
            set(audit),
            {"train", "val", "expert_min1_100agree", "expert_min2_100agree", "expert_min3_100agree"},
        )
        # ensure_writable_roots() creates the top-level runs/checkpoints scaffolding
        # even during --validate-only, but RunLogger.start() must never be reached,
        # so no run-specific metadata.json should exist anywhere under output_root.
        self.assertEqual(list(self.output_root.rglob("metadata.json")), [])

    def test_full_level_succeeds_when_dataset_files_match(self) -> None:
        for source_id in ("A", "B", "C1", "C2", "C3"):
            self._write_real_files(source_id)

        audit = self._run_main(validation_level="full")

        self.assertIn("expert_min3_100agree", audit)

    def test_full_level_raises_when_expert_split_file_missing(self) -> None:
        for source_id in ("A", "B", "C1", "C2"):
            self._write_real_files(source_id)
        # expert_min3's referenced files are intentionally never written.

        with self.assertRaises(FileNotFoundError):
            self._run_main(validation_level="full")


class ExpertManifestDecouplingRegressionTest(unittest.TestCase):
    def test_training_pairs_are_built_only_from_train_and_val_manifests(self) -> None:
        source = inspect.getsource(paper_train.main)
        pairs_line = next(
            line for line in source.splitlines() if "pairs = {name: load_pairs_from_manifest" in line
        )
        self.assertIn("manifests.items()", pairs_line)
        self.assertNotIn("audit_manifests.items()", pairs_line)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _fake_per_class() -> list[dict]:
    return [
        {
            "class_index": index,
            "support": 4,
            "predicted": 4,
            "true_positive": 3,
            "false_positive": 1,
            "false_negative": 1,
            "iou": 0.5,
            "dice_f1": 0.6,
            "precision": 0.75,
            "recall": 0.6,
        }
        for index in range(4)
    ]


class CheckpointSelectionTests(_ManifestFixtureMixin, unittest.TestCase):
    """Confirms last.pth is overwritten every epoch while best_val_miou.pth is
    only overwritten on epochs that actually improve mean_iou (Task #9)."""

    def test_best_checkpoint_only_updates_on_improving_epochs(self) -> None:
        self._write_real_files("A")
        self._write_real_files("B")
        self._manifest("train.csv", self._row("A", "SEQ_A"))
        self._manifest("val.csv", self._row("B", "SEQ_B"))
        self._manifest("expert_min1.csv", self._row("C1", "SEQ_C1", agreement="min1-100agree"))
        self._manifest("expert_min2.csv", self._row("C2", "SEQ_C2", agreement="min2-100agree"))
        self._manifest("expert_min3.csv", self._row("C3", "SEQ_C3", agreement="min3-100agree"))
        (self.manifest_root / "unused.csv").write_text("placeholder\n", encoding="utf-8")
        config_path = self.root / "config.yaml"
        config_path.write_text(
            "runtime:\n"
            "  run_id: checkpoint-selection-test\n"
            "data:\n"
            "  dataset_manifest: unused.csv\n"
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
            "  batch_size: 1\n"
            "  gradient_accumulation_steps: 1\n"
            "  epochs: 3\n"
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
        argv = [
            "paper_train.py",
            "--config", str(config_path),
            "--dataset-root", str(self.dataset_root),
            "--manifest-root", str(self.manifest_root),
            "--output-root", str(self.output_root),
        ]

        # mean_iou sequence: epoch 1 improves (from no prior best), epoch 2
        # improves again, epoch 3 regresses -- so best_val_miou.pth must stay
        # pinned to epoch 2 while last.pth advances to epoch 3.
        scripted_metrics = iter(
            [
                {"val_loss": 1.0, "pixel_accuracy": 0.5, "mean_iou": 0.4, "per_class": _fake_per_class(), "confusion_matrix": [[1] * 4] * 4},
                {"val_loss": 0.9, "pixel_accuracy": 0.55, "mean_iou": 0.6, "per_class": _fake_per_class(), "confusion_matrix": [[1] * 4] * 4},
                {"val_loss": 0.95, "pixel_accuracy": 0.5, "mean_iou": 0.5, "per_class": _fake_per_class(), "confusion_matrix": [[1] * 4] * 4},
            ]
        )

        def fake_evaluate(*args, **kwargs):
            return next(scripted_metrics)

        def fake_train_one_epoch(*args, **kwargs):
            return {"mean_loss": 0.1, "optimizer_steps": 1}

        with patch.object(sys, "argv", argv), \
                patch.object(paper_train, "build_deeplabv3plus", lambda spec: TinyModel()), \
                patch.object(paper_train, "evaluate", fake_evaluate), \
                patch.object(paper_train, "train_one_epoch", fake_train_one_epoch):
            paper_train.main()

        run_dir = self.output_root / "runs" / "checkpoint-selection-test"
        last_state = torch.load(run_dir / "checkpoints" / "last.pth", weights_only=False)
        best_state = torch.load(run_dir / "checkpoints" / "best_val_miou.pth", weights_only=False)
        self.assertEqual(last_state["epoch"], 3)
        self.assertEqual(best_state["epoch"], 2)
        self.assertEqual(best_state["best_validation_metric"], 0.6)


if __name__ == "__main__":
    unittest.main()
