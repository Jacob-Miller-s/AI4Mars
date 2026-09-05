import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from ai4mars import paper_train


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


class GlobalStepAccountingTests(unittest.TestCase):
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

    def _manifest(self, name: str, row: dict[str, str]) -> None:
        path = self.manifest_root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def _write_real_files(self, source_id: str) -> None:
        images_dir = self.dataset_root / "msl" / "ncam" / "images"
        labels_dir = self.dataset_root / "msl" / "ncam" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images_dir / f"{source_id}.JPG")
        mask = np.zeros((8, 8), dtype=np.uint8)
        Image.fromarray(mask).save(labels_dir / f"{source_id}.png")

    def test_global_step_uses_optimizer_steps_from_train_result(self) -> None:
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
            "  run_id: global-step-accounting-test\n"
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
            "  batch_size: 2\n"
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

        scripted_metrics = iter(
            [
                {"val_loss": 1.0, "pixel_accuracy": 0.5, "mean_iou": 0.4, "per_class": _fake_per_class(), "confusion_matrix": [[1] * 4] * 4},
                {"val_loss": 0.95, "pixel_accuracy": 0.51, "mean_iou": 0.45, "per_class": _fake_per_class(), "confusion_matrix": [[1] * 4] * 4},
                {"val_loss": 0.93, "pixel_accuracy": 0.52, "mean_iou": 0.46, "per_class": _fake_per_class(), "confusion_matrix": [[1] * 4] * 4},
            ]
        )
        scripted_train = iter(
            [
                {"mean_loss": 0.1, "optimizer_steps": 0},
                {"mean_loss": 0.1, "optimizer_steps": 1},
                {"mean_loss": 0.1, "optimizer_steps": 0},
            ]
        )

        def fake_evaluate(*args, **kwargs):
            return next(scripted_metrics)

        def fake_train_one_epoch(*args, **kwargs):
            return next(scripted_train)

        with patch.object(sys, "argv", argv), \
                patch.object(paper_train, "build_deeplabv3plus", lambda spec: TinyModel()), \
                patch.object(paper_train, "evaluate", fake_evaluate), \
                patch.object(paper_train, "train_one_epoch", fake_train_one_epoch):
            paper_train.main()

        run_dir = self.output_root / "runs" / "global-step-accounting-test"
        last_state = torch.load(run_dir / "checkpoints" / "last.pth", weights_only=False)
        self.assertEqual(last_state["global_step"], 1)


if __name__ == "__main__":
    unittest.main()
