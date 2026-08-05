"""Tests for src/paper_evaluate.py: the sole entry point that scores expert splits."""

import csv
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.paper_evaluate import (
    _write_confusion_matrix_csv,
    _write_confusion_matrix_figure,
    _write_per_class_csv,
    evaluate_split,
    load_frozen_checkpoint,
)
from src.train_utils import save_checkpoint


class TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TinySegmentationDataset(Dataset):
    def __init__(self, num_samples=3, size=4):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = torch.rand(3, self.size, self.size)
        mask = torch.zeros(self.size, self.size, dtype=torch.long)
        # Give every class some support so per-class metrics are well-defined.
        mask[0, :] = 0
        mask[1, :] = 1
        mask[2, :] = 2
        mask[3, :] = 3
        return image, mask


class LoadFrozenCheckpointTests(unittest.TestCase):
    def test_restores_weights_and_sets_eval_mode(self):
        source_model = TinySegmentationModel()
        optimizer = torch.optim.SGD(source_model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "frozen.pth"
            save_checkpoint(
                source_model,
                optimizer,
                epoch=7,
                path=checkpoint_path,
                metadata={"manifest_hash": "abc123"},
            )

            target_model = TinySegmentationModel()
            target_model.train()
            provenance = load_frozen_checkpoint(target_model, checkpoint_path, torch.device("cpu"))

        for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
            self.assertTrue(torch.equal(source_param, target_param))
        self.assertFalse(target_model.training)
        self.assertEqual(provenance["source_epoch"], 7)
        self.assertEqual(provenance["manifest_hash"], "abc123")

    def test_missing_model_state_dict_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "broken.pth"
            torch.save({"epoch": 1}, checkpoint_path)

            model = TinySegmentationModel()
            with self.assertRaisesRegex(ValueError, "model_state_dict"):
                load_frozen_checkpoint(model, checkpoint_path, torch.device("cpu"))


class WriterHelperTests(unittest.TestCase):
    def test_write_per_class_csv_contents(self):
        per_class = [
            {
                "support": 10,
                "predicted": 8,
                "true_positive": 6,
                "false_positive": 2,
                "false_negative": 4,
                "iou": 0.5,
                "dice_f1": 0.6,
                "precision": 0.75,
                "recall": 0.6,
            }
            for _ in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "per_class.csv"
            _write_per_class_csv(path, per_class)
            self.assertTrue(path.exists())
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(rows[0][0], "class_name")
        self.assertEqual([row[0] for row in rows[1:]], ["soil", "bedrock", "sand", "big_rock"])
        self.assertEqual(rows[1][1:], ["10", "8", "6", "2", "4", "0.5", "0.6", "0.75", "0.6"])

    def test_write_confusion_matrix_csv_contents(self):
        matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "confusion.csv"
            _write_confusion_matrix_csv(path, matrix)
            self.assertTrue(path.exists())
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(rows[0][1:], ["soil", "bedrock", "sand", "big_rock"])
        self.assertEqual(rows[1], ["soil", "1", "2", "3", "4"])
        self.assertEqual(rows[4], ["big_rock", "13", "14", "15", "16"])

    def test_write_confusion_matrix_figure_creates_file(self):
        normalized = [[0.7, 0.1, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "confusion.png"
            _write_confusion_matrix_figure(path, normalized, "expert_min1")
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


class EvaluateSplitTests(unittest.TestCase):
    def test_returns_detailed_metrics_with_per_class_and_confusion_matrix(self):
        model = TinySegmentationModel()
        loader = DataLoader(TinySegmentationDataset(), batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        metrics = evaluate_split(model, loader, loss_fn, torch.device("cpu"))

        for key in ("val_loss", "pixel_accuracy", "mean_iou", "per_class", "confusion_matrix", "normalized_confusion_matrix"):
            self.assertIn(key, metrics)
        self.assertEqual(len(metrics["per_class"]), 4)
        self.assertEqual(len(metrics["confusion_matrix"]), 4)


if __name__ == "__main__":
    unittest.main()
