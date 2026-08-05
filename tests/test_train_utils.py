import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train_utils import evaluate


class FixedClassZeroModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = images.shape
        logits = torch.zeros((batch_size, 4, height, width), device=images.device)
        logits[:, 0] = 1.0
        return logits


class EvaluateTests(unittest.TestCase):
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
        self.assertEqual(results["pixel_accuracy"], 1.0)

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


if __name__ == "__main__":
    unittest.main()