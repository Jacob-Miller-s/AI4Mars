import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.paper_train import _IndexedDataset
from src.train_utils import evaluate, train_one_epoch


class FixedClassZeroModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = images.shape
        logits = torch.zeros((batch_size, 4, height, width), device=images.device)
        logits[:, 0] = 1.0
        return logits


class _DiagnosticLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_training_diagnostic(self, *, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))


class _ThreeItemDataset(torch.utils.data.Dataset):
    def __init__(self, masks: list[torch.Tensor], *, prefix: str = "SRC") -> None:
        self._images = [torch.zeros((3, 4, 4), dtype=torch.float32) for _ in masks]
        self._masks = masks
        self._prefix = prefix

    def __len__(self) -> int:
        return len(self._masks)

    def __getitem__(self, index: int):
        return self._images[index], self._masks[index], f"{self._prefix}_{index}"


class _InvalidTupleDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        del index
        image = torch.zeros((3, 4, 4), dtype=torch.float32)
        mask = torch.zeros((4, 4), dtype=torch.long)
        return image, mask, "SRC_0", "EXTRA"


class _NanLoss(nn.Module):
    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        del logits, masks
        return torch.tensor(float("nan"))


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


class TrainBatchContractTests(unittest.TestCase):
    def test_two_element_batch_contract_trains(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
        masks = torch.zeros((2, 4, 4), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)

        result = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"), gradient_accumulation_steps=1)

        self.assertEqual(result["processed_batches"], 1)
        self.assertEqual(result["skipped_all_ignore_batches"], 0)
        self.assertEqual(result["optimizer_steps"], 1)

    def test_three_element_batch_contract_trains(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        masks = [torch.zeros((4, 4), dtype=torch.long), torch.ones((4, 4), dtype=torch.long)]
        loader = DataLoader(_ThreeItemDataset(masks), batch_size=2, shuffle=False)

        result = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"), gradient_accumulation_steps=1)

        self.assertEqual(result["processed_batches"], 1)
        self.assertEqual(result["skipped_all_ignore_batches"], 0)

    def test_source_ids_survive_dataloader_collation(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        logger = _DiagnosticLogger()
        masks = [torch.full((4, 4), 255, dtype=torch.long), torch.full((4, 4), 255, dtype=torch.long)]
        loader = DataLoader(_ThreeItemDataset(masks), batch_size=2, shuffle=False)

        with self.assertRaisesRegex(RuntimeError, "all batches were all-ignore and skipped"):
            train_one_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                torch.device("cpu"),
                epoch=2,
                run_logger=logger,
            )

        self.assertEqual(logger.events[0][0], "all_ignore_batch")
        self.assertEqual(logger.events[0][1]["sample_ids"], ["SRC_0", "SRC_1"])

    def test_unsupported_tuple_lengths_fail_clearly(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        loader = DataLoader(_InvalidTupleDataset(), batch_size=1, shuffle=False)

        with self.assertRaisesRegex(ValueError, "exactly 2 or 3 items"):
            train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

    def test_all_ignore_three_element_batch_is_skipped_and_params_unchanged(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        initial = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        logger = _DiagnosticLogger()
        masks = [torch.full((4, 4), 255, dtype=torch.long), torch.full((4, 4), 255, dtype=torch.long)]
        loader = DataLoader(_ThreeItemDataset(masks), batch_size=2, shuffle=False)

        with self.assertRaisesRegex(RuntimeError, "all batches were all-ignore and skipped"):
            train_one_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                torch.device("cpu"),
                epoch=1,
                run_logger=logger,
            )

        self.assertTrue(torch.equal(model.weight.detach(), initial))
        self.assertEqual(logger.events[0][0], "all_ignore_batch")

    def test_mixed_valid_and_all_ignore_sample_batch_trains_normally(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        masks = [torch.zeros((4, 4), dtype=torch.long), torch.full((4, 4), 255, dtype=torch.long)]
        loader = DataLoader(_ThreeItemDataset(masks), batch_size=2, shuffle=False)

        result = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

        self.assertEqual(result["processed_batches"], 1)
        self.assertEqual(result["skipped_all_ignore_batches"], 0)
        self.assertEqual(result["optimizer_steps"], 1)

    def test_skipped_batches_do_not_advance_gradient_accumulation(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        masks = [
            torch.zeros((4, 4), dtype=torch.long),
            torch.full((4, 4), 255, dtype=torch.long),
            torch.ones((4, 4), dtype=torch.long),
        ]
        loader = DataLoader(_ThreeItemDataset(masks), batch_size=1, shuffle=False)

        result = train_one_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            torch.device("cpu"),
            gradient_accumulation_steps=2,
        )

        self.assertEqual(result["processed_batches"], 2)
        self.assertEqual(result["skipped_all_ignore_batches"], 1)
        self.assertEqual(result["optimizer_steps"], 1)

    def test_mean_loss_excludes_skipped_batches(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        image = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        valid_mask = torch.zeros((1, 4, 4), dtype=torch.long)
        ignore_mask = torch.full((1, 4, 4), 255, dtype=torch.long)

        with torch.no_grad():
            expected = float(loss_fn(model(image), valid_mask).item())

        images = torch.cat([image, image], dim=0)
        masks = torch.cat([valid_mask, ignore_mask], dim=0)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)
        result = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

        self.assertEqual(result["processed_batches"], 1)
        self.assertEqual(result["skipped_all_ignore_batches"], 1)
        self.assertAlmostEqual(result["mean_loss"], expected, places=6)

    def test_source_ids_appear_in_fatal_diagnostics(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        logger = _DiagnosticLogger()
        masks = [torch.zeros((4, 4), dtype=torch.long)]
        loader = DataLoader(_ThreeItemDataset(masks, prefix="S"), batch_size=1, shuffle=False)

        with self.assertRaisesRegex(RuntimeError, "S_0"):
            train_one_epoch(
                model,
                loader,
                optimizer,
                _NanLoss(),
                torch.device("cpu"),
                epoch=7,
                run_logger=logger,
            )

        self.assertEqual(logger.events[0][0], "non_finite_loss")
        self.assertEqual(logger.events[0][1]["sample_ids"], ["S_0"])

    def test_indexed_dataset_and_train_one_epoch_integration(self) -> None:
        class _BaseDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int):
                del index
                return torch.zeros((3, 4, 4), dtype=torch.float32), torch.zeros((4, 4), dtype=torch.long)

        base = _BaseDataset()
        indexed = _IndexedDataset(base, ["A", "B"])
        loader = DataLoader(indexed, batch_size=2, shuffle=False)
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        result = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

        self.assertEqual(result["processed_batches"], 1)
        self.assertEqual(result["skipped_all_ignore_batches"], 0)


if __name__ == "__main__":
    unittest.main()
