import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai4mars.paper_train import _IndexedDataset
from ai4mars.train_utils import evaluate, train_one_epoch


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


class _NanLogitModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = images.shape
        del images
        return torch.full((batch_size, 4, height, width), float("nan"))


class _FakeScaledLoss:
    def __init__(self, loss: torch.Tensor, scale: float) -> None:
        self.loss = loss
        self.scale = float(scale)

    def backward(self) -> None:
        (self.loss * self.scale).backward()


class _FakeGradScaler:
    def __init__(self, *, initial_scale: float = 8.0, overflow_unscale_calls: set[int] | None = None) -> None:
        self.current_scale = float(initial_scale)
        self.overflow_unscale_calls = set() if overflow_unscale_calls is None else set(overflow_unscale_calls)
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0
        self._overflow_active = False

    def get_scale(self) -> float:
        return self.current_scale

    def scale(self, loss: torch.Tensor) -> _FakeScaledLoss:
        return _FakeScaledLoss(loss, self.current_scale)

    def unscale_(self, optimizer) -> None:
        self.unscale_calls += 1
        overflow_now = self.unscale_calls in self.overflow_unscale_calls
        self._overflow_active = overflow_now
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.grad.data = parameter.grad.data / self.current_scale
                if overflow_now:
                    parameter.grad.data.fill_(float("inf"))

    def step(self, optimizer) -> None:
        self.step_calls += 1
        if self._overflow_active:
            return
        optimizer.step()

    def update(self) -> None:
        self.update_calls += 1
        if self._overflow_active:
            self.current_scale = max(self.current_scale / 2.0, 1e-20)
        self._overflow_active = False


@contextmanager
def _cpu_amp_patch():
    with patch("torch.Tensor.to", new=lambda self, *args, **kwargs: self):
        yield


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


class TrainAmpOverflowBehaviorTests(unittest.TestCase):
    def test_amp_unscales_only_at_accumulation_boundary(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(initial_scale=16.0)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((2, 3, 4, 4), dtype=torch.float32)
        masks = torch.zeros((2, 4, 4), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)

        with _cpu_amp_patch():
            result = train_one_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                torch.device("cuda"),
                amp_enabled=True,
                scaler=scaler,
                gradient_accumulation_steps=2,
            )

        self.assertEqual(result["processed_batches"], 2)
        self.assertEqual(scaler.unscale_calls, 1)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)

    def test_recoverable_amp_overflow_skips_optimizer_update(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        initial = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(initial_scale=8.0, overflow_unscale_calls={1})
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((1, 3, 4, 4), dtype=torch.float32)
        masks = torch.zeros((1, 4, 4), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)

        with _cpu_amp_patch():
            result = train_one_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                torch.device("cuda"),
                amp_enabled=True,
                scaler=scaler,
                max_consecutive_amp_overflow_steps=4,
            )

        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["skipped_amp_overflow_steps"], 1)
        self.assertAlmostEqual(result["minimum_amp_scale"], 4.0)
        self.assertTrue(torch.equal(initial, model.weight.detach()))
        self.assertEqual(scaler.update_calls, 1)

    def test_training_continues_after_single_overflow(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(initial_scale=8.0, overflow_unscale_calls={1})
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((3, 3, 4, 4), dtype=torch.float32)
        masks = torch.zeros((3, 4, 4), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)

        with _cpu_amp_patch():
            result = train_one_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                torch.device("cuda"),
                amp_enabled=True,
                scaler=scaler,
            )

        self.assertEqual(result["processed_batches"], 3)
        self.assertEqual(result["skipped_amp_overflow_steps"], 1)
        self.assertEqual(result["optimizer_steps"], 2)

    def test_final_partial_accumulation_uses_same_amp_finalize_logic(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(initial_scale=8.0, overflow_unscale_calls={2})
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((3, 3, 4, 4), dtype=torch.float32)
        masks = torch.zeros((3, 4, 4), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)

        with _cpu_amp_patch():
            result = train_one_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                torch.device("cuda"),
                amp_enabled=True,
                scaler=scaler,
                gradient_accumulation_steps=2,
            )

        self.assertEqual(result["processed_batches"], 3)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["skipped_amp_overflow_steps"], 1)
        self.assertEqual(scaler.unscale_calls, 2)

    def test_repeated_overflow_protection_fails_clearly(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = _FakeGradScaler(initial_scale=8.0, overflow_unscale_calls={1, 2})
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((2, 3, 4, 4), dtype=torch.float32)
        masks = torch.zeros((2, 4, 4), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)

        with _cpu_amp_patch():
            with self.assertRaisesRegex(RuntimeError, "AMP overflow skipped too many consecutive optimizer steps"):
                train_one_epoch(
                    model,
                    loader,
                    optimizer,
                    loss_fn,
                    torch.device("cuda"),
                    amp_enabled=True,
                    scaler=scaler,
                    max_consecutive_amp_overflow_steps=2,
                )

    def test_non_finite_gradients_without_amp_remain_fatal(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        images = torch.randn((1, 3, 4, 4), dtype=torch.float32)
        masks = torch.zeros((1, 4, 4), dtype=torch.long)
        loader = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)

        with patch("ai4mars.train_utils._gradients_are_finite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Training gradients became non-finite"):
                train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))

    def test_non_finite_inputs_and_logits_remain_fatal(self) -> None:
        model = nn.Conv2d(3, 4, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        images = torch.full((1, 3, 4, 4), float("nan"), dtype=torch.float32)
        masks = torch.zeros((1, 4, 4), dtype=torch.long)
        loader_inputs = DataLoader(TensorDataset(images, masks), batch_size=1, shuffle=False)
        with self.assertRaisesRegex(RuntimeError, "inputs are non-finite"):
            train_one_epoch(model, loader_inputs, optimizer, loss_fn, torch.device("cpu"))

        clean_images = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        loader_logits = DataLoader(TensorDataset(clean_images, masks), batch_size=1, shuffle=False)
        with self.assertRaisesRegex(RuntimeError, "logits are non-finite"):
            train_one_epoch(
                _NanLogitModel(),
                dataloader=loader_logits,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
