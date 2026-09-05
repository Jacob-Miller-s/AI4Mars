import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from ai4mars.train_utils import load_training_checkpoint, restore_rng_state, save_checkpoint


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "checkpoint.pth"
        self.metadata = {"manifest_hash": "manifest", "git_commit_sha": "first"}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_round_trip_restores_optimizer_and_resume_state(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        expected_weight = model.weight.detach().clone()
        save_checkpoint(model, optimizer, 3, self.path, self.metadata, scheduler=scheduler, global_step=12, best_validation_metric=0.6)

        restored_model = nn.Linear(2, 1)
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
        state = load_training_checkpoint(restored_model, restored_optimizer, self.path, torch.device("cpu"), scheduler=restored_scheduler, expected_metadata=self.metadata)

        self.assertTrue(torch.equal(restored_model.weight, expected_weight))
        self.assertEqual(state["epoch"], 3)
        self.assertEqual(state["global_step"], 12)
        self.assertEqual(state["best_validation_metric"], 0.6)

    def test_manifest_mismatch_refuses_resume_but_commit_difference_warns(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(model, optimizer, 1, self.path, self.metadata)

        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            load_training_checkpoint(model, optimizer, self.path, torch.device("cpu"), expected_metadata={"manifest_hash": "other"})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            load_training_checkpoint(model, optimizer, self.path, torch.device("cpu"), expected_metadata={"manifest_hash": "manifest", "git_commit_sha": "second"})
        self.assertTrue(any("source commit differs" in str(item.message) for item in captured))

    def test_resume_ignores_only_operational_resume_checkpoint_path(self) -> None:
        checkpoint_metadata = {
            "configuration": {
                "training": {"resume_checkpoint": "/previous/last.pth", "seed": 42},
                "model": {"backbone": "resnet101"},
            }
        }
        expected_metadata = {
            "configuration": {
                "training": {"resume_checkpoint": "/new/location/last.pth", "seed": 42},
                "model": {"backbone": "resnet101"},
            }
        }
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(model, optimizer, 1, self.path, checkpoint_metadata)

        load_training_checkpoint(model, optimizer, self.path, torch.device("cpu"), expected_metadata=expected_metadata)

        incompatible = {
            "configuration": {
                "training": {"resume_checkpoint": "/new/location/last.pth", "seed": 99},
                "model": {"backbone": "resnet101"},
            }
        }
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            load_training_checkpoint(model, optimizer, self.path, torch.device("cpu"), expected_metadata=incompatible)

    def test_rng_states_are_normalized_after_cpu_checkpoint_load(self) -> None:
        torch_state = torch.arange(16, dtype=torch.int64)[::2]
        cuda_state = torch.arange(12, dtype=torch.int16)[::2]
        with (
            patch("ai4mars.train_utils.torch.set_rng_state") as set_torch_state,
            patch("ai4mars.train_utils.torch.cuda.is_available", return_value=True),
            patch("ai4mars.train_utils.torch.cuda.set_rng_state_all") as set_cuda_states,
        ):
            restore_rng_state({"torch": torch_state, "cuda": [cuda_state]})

        restored_torch = set_torch_state.call_args.args[0]
        restored_cuda = set_cuda_states.call_args.args[0][0]
        self.assertEqual((restored_torch.device.type, restored_torch.dtype, restored_torch.is_contiguous()), ("cpu", torch.uint8, True))
        self.assertEqual((restored_cuda.device.type, restored_cuda.dtype, restored_cuda.is_contiguous()), ("cpu", torch.uint8, True))

    def test_amp_is_rejected_on_cpu(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset
        from ai4mars.train_utils import train_one_epoch

        model = nn.Conv2d(3, 4, kernel_size=1)
        loader = DataLoader(TensorDataset(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(), torch.device("cpu"), amp_enabled=True)


if __name__ == "__main__":
    unittest.main()