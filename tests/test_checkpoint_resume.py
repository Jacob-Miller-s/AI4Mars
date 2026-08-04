import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from src.train_utils import load_training_checkpoint, save_checkpoint


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

    def test_amp_is_rejected_on_cpu(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset
        from src.train_utils import train_one_epoch

        model = nn.Conv2d(3, 4, kernel_size=1)
        loader = DataLoader(TensorDataset(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(), torch.device("cpu"), amp_enabled=True)


if __name__ == "__main__":
    unittest.main()