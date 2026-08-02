import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train_utils import _cuda_allocator_metrics, evaluate, train_one_epoch


class FixedClassZeroModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = images.shape
        logits = torch.zeros((batch_size, 4, height, width), device=images.device)
        logits[:, 0] = 1.0
        return logits


class RecordingLogger:
    def __init__(self) -> None:
        self.batch_events = []
        self.epoch_events = []

    def log_batch(self, **kwargs) -> None:
        self.batch_events.append(kwargs)

    def log_epoch(self, event) -> None:
        self.epoch_events.append(event)


class EvaluateTests(unittest.TestCase):
    def test_cuda_allocator_metrics_require_initialized_cuda(self) -> None:
        with patch("src.train_utils.torch.cuda.is_initialized", return_value=False) as initialized, patch(
            "src.train_utils.torch.cuda.memory_allocated"
        ) as allocated:
            self.assertEqual(_cuda_allocator_metrics(torch.device("cpu")), {})
            self.assertEqual(_cuda_allocator_metrics(torch.device("cuda")), {})

        initialized.assert_called_once()
        allocated.assert_not_called()

        with patch("src.train_utils.torch.cuda.is_initialized", return_value=True), patch(
            "src.train_utils.torch.cuda.memory_allocated", return_value=64
        ) as allocated, patch("src.train_utils.torch.cuda.memory_reserved", return_value=128) as reserved:
            metrics = _cuda_allocator_metrics(torch.device("cuda"))

        self.assertEqual(metrics, {"gpu_memory_allocated_bytes": 64, "gpu_memory_reserved_bytes": 128})
        allocated.assert_called_once()
        reserved.assert_called_once()

    def test_skips_all_ignore_batches_when_averaging_validation_loss(self) -> None:
        images = torch.zeros((2, 3, 2, 2))
        masks = torch.tensor(
            [
                [[255, 255], [255, 255]],
                [[0, 0], [0, 0]],
            ]
        )
        dataloader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)

        results = evaluate(
            FixedClassZeroModel(),
            dataloader,
            nn.CrossEntropyLoss(ignore_index=255),
            torch.device("cpu"),
        )

        self.assertTrue(torch.isfinite(torch.tensor(results["val_loss"])))
        self.assertEqual(results["finite_loss_batches"], 1)
        self.assertEqual(results["skipped_all_ignore_loss_batches"], 1)
        self.assertEqual(results["pixel_acc"], 1.0)

    def test_rejects_evaluation_split_without_valid_target_pixels(self) -> None:
        images = torch.zeros((1, 3, 2, 2))
        masks = torch.full((1, 2, 2), 255, dtype=torch.long)
        dataloader = DataLoader(TensorDataset(images, masks), batch_size=1)

        with self.assertRaisesRegex(RuntimeError, "no batches with valid target pixels"):
            evaluate(
                FixedClassZeroModel(),
                dataloader,
                nn.CrossEntropyLoss(ignore_index=255),
                torch.device("cpu"),
            )

    def test_training_and_evaluation_emit_optional_console_events(self) -> None:
        images = torch.zeros((2, 3, 2, 2))
        masks = torch.zeros((2, 2, 2), dtype=torch.long)
        dataloader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        logger = RecordingLogger()

        train_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            nn.CrossEntropyLoss(ignore_index=255),
            torch.device("cpu"),
            epoch=1,
            run_logger=logger,
            batch_log_interval=1,
        )
        results = evaluate(
            model,
            dataloader,
            nn.CrossEntropyLoss(ignore_index=255),
            torch.device("cpu"),
            return_detailed_metrics=True,
            epoch=1,
            train_loss=train_loss,
            run_logger=logger,
        )

        self.assertEqual(len(logger.batch_events), 2)
        self.assertEqual(len(logger.epoch_events), 1)
        self.assertIn("normalized_confusion_matrix", results)
        self.assertEqual(logger.epoch_events[0].epoch, 1)


if __name__ == "__main__":
    unittest.main()