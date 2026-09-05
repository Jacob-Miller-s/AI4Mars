import unittest

import torch

from ai4mars.metrics import segmentation_confusion_matrix


class SegmentationConfusionMatrixTests(unittest.TestCase):
    def test_uses_ground_truth_rows_prediction_columns_and_ignores_masked_pixels(self) -> None:
        predictions = torch.tensor([[[0, 1, 3], [2, 0, 1]]])
        targets = torch.tensor([[[0, 1, 2], [3, 255, 1]]])

        actual = segmentation_confusion_matrix(predictions, targets, num_classes=4)

        expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_returns_empty_matrix_when_every_target_pixel_is_ignored(self) -> None:
        predictions = torch.tensor([[[0, 1], [2, 3]]])
        targets = torch.full((1, 2, 2), 255)

        actual = segmentation_confusion_matrix(predictions, targets, num_classes=4)

        self.assertTrue(torch.equal(actual, torch.zeros((4, 4), dtype=torch.long)))

    def test_rejects_valid_labels_outside_the_class_range(self) -> None:
        predictions = torch.tensor([[[0, 4]]])
        targets = torch.tensor([[[0, 1]]])

        with self.assertRaisesRegex(ValueError, "must be in"):
            segmentation_confusion_matrix(predictions, targets, num_classes=4)


if __name__ == "__main__":
    unittest.main()