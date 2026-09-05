import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from ai4mars.foundation import sha256_file
from ai4mars.reproduction import (
    VerifiedCheckpoint,
    acquire_frozen_checkpoint,
    load_onboarding_samples,
    run_onboarding,
    verify_frozen_checkpoint,
)


class ReproductionInterfaceTests(unittest.TestCase):
    def test_acquire_frozen_checkpoint_reuses_a_verified_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pth"
            torch.save({"epoch": 25, "model_state_dict": {}}, checkpoint_path)

            verified = acquire_frozen_checkpoint(
                checkpoint_path,
                source_url="https://invalid.example/checkpoint.pth",
                expected_sha256=sha256_file(checkpoint_path),
            )

            self.assertEqual(verified.path, checkpoint_path)
            self.assertEqual(verified.epoch, 25)

    def test_committed_onboarding_sample_is_complete_and_verified(self) -> None:
        sample_root = Path(__file__).parents[1] / "data" / "samples" / "onboarding"

        samples = load_onboarding_samples(sample_root)

        self.assertEqual(len(samples), 8)
        self.assertEqual(len({sample.sequence_id for sample in samples}), 8)
        self.assertEqual(
            {label for sample in samples for label, count in sample.class_counts.items() if label in range(4) and count},
            {0, 1, 2, 3},
        )

    def test_run_onboarding_returns_verified_predictions_and_metrics(self) -> None:
        class ConstantModel(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.zeros(
                    (images.shape[0], 4, images.shape[2], images.shape[3]),
                    dtype=images.dtype,
                    device=images.device,
                )

        project_root = Path(__file__).parents[1]
        verified = VerifiedCheckpoint(
            path=project_root / "checkpoint.pth",
            sha256="a" * 64,
            epoch=25,
            payload={"model_state_dict": {}},
        )
        with (
            patch("ai4mars.reproduction.verify_frozen_checkpoint", return_value=verified),
            patch("ai4mars.reproduction.build_deeplabv3plus", return_value=ConstantModel()),
        ):
            report = run_onboarding(
                config_path=project_root / "configs" / "reproduction" / "paper_deeplabv3plus_kaggle_p100.yaml",
                checkpoint_path=verified.path,
                sample_root=project_root / "data" / "samples" / "onboarding",
                device="cpu",
                expected_metric_ranges={"pixel_accuracy": (0.0, 1.0), "mean_iou": (0.0, 1.0)},
            )

        self.assertEqual(report.device, "cpu")
        self.assertEqual(report.checkpoint_epoch, 25)
        self.assertEqual(len(report.predictions), 8)
        self.assertEqual(report.predictions[0].prediction.shape, (513, 513))
        self.assertEqual(len(report.metrics["per_class"]), 4)

    def test_verify_frozen_checkpoint_enforces_hash_epoch_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pth"
            torch.save({"epoch": 25, "model_state_dict": {"weight": torch.ones(1)}}, checkpoint_path)

            verified = verify_frozen_checkpoint(
                checkpoint_path,
                expected_sha256=sha256_file(checkpoint_path),
            )

            self.assertEqual(verified.epoch, 25)
            self.assertEqual(verified.path, checkpoint_path)
            self.assertIn("model_state_dict", verified.payload)

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_frozen_checkpoint(checkpoint_path, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()