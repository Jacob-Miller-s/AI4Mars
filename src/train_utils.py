"""
src/train_utils.py
==================
Reusable training utilities for the AI4Mars segmentation pipeline.

Tensor shape conventions used throughout:
    - Input images  : [B, 3, H, W]  float32
    - Output logits : [B, num_classes, H, W]  float32  (raw, un-softmaxed)
    - Target masks  : [B, H, W]  int64 (long)

Loss function:
    torch.nn.CrossEntropyLoss(ignore_index=255)

    CrossEntropyLoss expects logits (NOT softmax probabilities) and target
    class IDs as integers.  The ignore_index=255 argument tells it to skip
    pixels labelled 255 (unlabeled / out-of-scope regions in AI4Mars masks).
"""

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional
import tempfile
import warnings

import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import segmentation_confusion_matrix, segmentation_metrics_from_confusion_matrix

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns
    -------
    torch.device
        The device object to pass to ``.to(device)``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    global_step: int = 0,
    best_validation_metric: Optional[float] = None,
    include_rng_state: bool = True,
) -> None:
    """Save model and optimizer state to disk.

    Parameters
    ----------
    model : nn.Module
        The model whose weights we want to save.
    optimizer : torch.optim.Optimizer
        The optimizer whose state we also save (allows resuming training).
    epoch : int
        Current epoch number (stored as metadata in the checkpoint).
    path : Path
        Destination file path — should end in ``.pth``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    payload["global_step"] = global_step
    payload["best_validation_metric"] = best_validation_metric
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if include_rng_state:
        payload["rng_state"] = capture_rng_state()

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
    expected_metadata: Optional[Dict[str, Any]] = None,
    require_metadata_match: bool = True,
) -> int:
    """Load model and optimizer weights from a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        Model to load weights into (must have the same architecture).
    optimizer : torch.optim.Optimizer
        Optimizer to restore state into.
    path : Path
        Path to the ``.pth`` checkpoint file.
    device : torch.device
        Device to map the tensors to when loading.

    Returns
    -------
    int
        The epoch at which the checkpoint was saved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if expected_metadata is not None:
        _validate_checkpoint_metadata(
            checkpoint=checkpoint,
            expected_metadata=expected_metadata,
            require_metadata_match=require_metadata_match,
            checkpoint_path=path,
        )

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {path}  (epoch {epoch})")
    return epoch


def capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state needed to resume a single-process training run."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _normalize_rng_tensor(value: Any, *, state_name: str) -> torch.Tensor:
    """Return a CPU-contiguous uint8 tensor accepted by PyTorch RNG setters."""
    try:
        tensor = torch.as_tensor(value, dtype=torch.uint8, device="cpu")
    except (TypeError, RuntimeError) as error:
        raise TypeError(f"Invalid {state_name} RNG state in checkpoint.") from error
    return tensor.contiguous()


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore previously captured single-process RNG state."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_normalize_rng_tensor(state["torch"], state_name="torch"))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("Invalid CUDA RNG states in checkpoint; expected a sequence.")
        torch.cuda.set_rng_state_all(
            [_normalize_rng_tensor(value, state_name="cuda") for value in cuda_states]
        )


def _normalize_resume_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the only operational field that must not define an experiment."""
    normalized = deepcopy(metadata)
    configuration = normalized.get("configuration")
    if isinstance(configuration, dict):
        training = configuration.get("training")
        if isinstance(training, dict):
            training.pop("resume_checkpoint", None)
    return normalized


def validate_resume_metadata(checkpoint_metadata: Dict[str, Any], expected_metadata: Dict[str, Any]) -> None:
    """Reject changed experiment definitions while allowing a source SHA change."""
    checkpoint_metadata = _normalize_resume_metadata(checkpoint_metadata)
    expected_metadata = _normalize_resume_metadata(expected_metadata)
    mismatches = []
    for key, expected_value in expected_metadata.items():
        actual_value = checkpoint_metadata.get(key)
        if actual_value == expected_value:
            continue
        if key == "git_commit_sha":
            warnings.warn(
                f"Checkpoint source commit differs: expected={expected_value!r}, actual={actual_value!r}",
                stacklevel=2,
            )
            continue
        mismatches.append((key, expected_value, actual_value))
    if mismatches:
        details = "; ".join(f"{key}: expected={expected!r}, actual={actual!r}" for key, expected, actual in mismatches)
        raise RuntimeError(f"Checkpoint is incompatible with this experiment: {details}")


def load_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
    *,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Restore all resumable state and return completed-epoch metadata."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if expected_metadata is not None:
        metadata = checkpoint.get("metadata")
        if metadata is None:
            raise RuntimeError("Checkpoint metadata missing; refusing an unsafe resume.")
        validate_resume_metadata(metadata, expected_metadata)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint.get("rng_state", {}))
    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_validation_metric": checkpoint.get("best_validation_metric"),
        "metadata": checkpoint.get("metadata", {}),
    }


def _validate_checkpoint_metadata(
    checkpoint: Dict[str, Any],
    expected_metadata: Dict[str, Any],
    require_metadata_match: bool,
    checkpoint_path: Path,
) -> None:
    """Validate selected metadata keys against expected values."""
    checkpoint_metadata = checkpoint.get("metadata")
    if checkpoint_metadata is None:
        message = (
            "Checkpoint metadata not found. Cannot verify split provenance for "
            f"{checkpoint_path}."
        )
        if require_metadata_match:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return

    mismatches = []
    for key, expected_value in expected_metadata.items():
        actual_value = checkpoint_metadata.get(key)
        if actual_value != expected_value:
            mismatches.append((key, expected_value, actual_value))

    if mismatches:
        mismatch_text = "; ".join(
            f"{key}: expected={expected!r}, actual={actual!r}"
            for key, expected, actual in mismatches
        )
        message = (
            "Checkpoint metadata does not match expected evaluation metadata: "
            f"{mismatch_text}"
        )
        if require_metadata_match:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


def _cuda_allocator_metrics(device: torch.device) -> Dict[str, int]:
    """Return allocator metrics only when the training process already uses CUDA."""
    if device.type != "cuda" or not torch.cuda.is_initialized():
        return {}
    try:
        return {
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
    except RuntimeError:
        return {}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _normalize_sample_ids(sample_ids: Any) -> Optional[list[str]]:
    if sample_ids is None:
        return None
    if isinstance(sample_ids, str):
        return [sample_ids]
    if isinstance(sample_ids, (list, tuple)):
        return [str(item) for item in sample_ids]
    return [str(sample_ids)]


def _unpack_training_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[list[str]]]:
    if not isinstance(batch, (tuple, list)):
        raise TypeError(
            "Training dataloader must yield a tuple/list batch with 2 items "
            "(images, masks) or 3 items (images, masks, sample_ids)."
        )
    if len(batch) == 2:
        images, masks = batch
        sample_ids = None
    elif len(batch) == 3:
        images, masks, sample_ids = batch
    else:
        raise ValueError(
            "Training dataloader batch must contain exactly 2 or 3 items: "
            f"received {len(batch)} items."
        )
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Batch images must be a torch.Tensor, got {type(images).__name__}.")
    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"Batch masks must be a torch.Tensor, got {type(masks).__name__}.")
    return images, masks, _normalize_sample_ids(sample_ids)


def _emit_training_diagnostic(run_logger: Optional[Any], *, event_type: str, payload: Dict[str, Any]) -> None:
    if run_logger is None:
        return
    method = getattr(run_logger, "log_training_diagnostic", None)
    if callable(method):
        method(event_type=event_type, **payload)


def _gradients_are_finite(model: nn.Module) -> bool:
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


def _finalize_optimizer_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    amp_enabled: bool,
    scaler: Optional[Any],
    run_logger: Optional[Any],
    epoch: Optional[int],
    accumulation_end_batch: int,
    accumulation_sample_ids: list[str],
    accumulation_supervised_batches: int,
) -> Dict[str, Any]:
    if amp_enabled:
        if scaler is None:
            raise ValueError("A CUDA GradScaler is required when AMP is enabled.")
        old_scale = float(scaler.get_scale())
        scaler.unscale_(optimizer)
        gradients_finite = _gradients_are_finite(model)
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        optimizer.zero_grad()
        if gradients_finite:
            return {
                "optimizer_step_taken": True,
                "amp_overflow_skipped": False,
                "old_scale": old_scale,
                "new_scale": new_scale,
            }
        _emit_training_diagnostic(
            run_logger,
            event_type="amp_overflow_step_skipped",
            payload={
                "epoch": epoch,
                "batch": accumulation_end_batch,
                "sample_ids": accumulation_sample_ids,
                "supervised_batches_in_window": accumulation_supervised_batches,
                "old_scale": old_scale,
                "new_scale": new_scale,
            },
        )
        return {
            "optimizer_step_taken": False,
            "amp_overflow_skipped": True,
            "old_scale": old_scale,
            "new_scale": new_scale,
        }

    if not _gradients_are_finite(model):
        diagnostic = {
            "epoch": epoch,
            "batch": accumulation_end_batch,
            "sample_ids": accumulation_sample_ids,
            "supervised_batches_in_window": accumulation_supervised_batches,
            "amp_enabled": False,
        }
        _emit_training_diagnostic(run_logger, event_type="non_finite_gradients", payload=diagnostic)
        raise RuntimeError(
            "Training gradients became non-finite for a batch with valid target pixels "
            f"(epoch={epoch}, batch={accumulation_end_batch}, sample_ids={accumulation_sample_ids})."
        )
    optimizer.step()
    optimizer.zero_grad()
    return {
        "optimizer_step_taken": True,
        "amp_overflow_skipped": False,
        "old_scale": None,
        "new_scale": None,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    epoch: Optional[int] = None,
    run_logger: Optional[Any] = None,
    batch_log_interval: int = 10,
    amp_enabled: bool = False,
    scaler: Optional[Any] = None,
    gradient_accumulation_steps: int = 1,
    ignore_index: Optional[int] = None,
    max_consecutive_amp_overflow_steps: int = 16,
    minimum_amp_scale_floor: float = 1e-12,
) -> Dict[str, Any]:
    """Run one full training epoch."""
    if batch_log_interval < 1:
        raise ValueError("batch_log_interval must be at least 1.")
    if run_logger is not None and epoch is None:
        raise ValueError("epoch is required when run_logger is provided.")
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP is supported only for CUDA training.")
    if amp_enabled and scaler is None:
        raise ValueError("A CUDA GradScaler is required when AMP is enabled.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    if max_consecutive_amp_overflow_steps < 1:
        raise ValueError("max_consecutive_amp_overflow_steps must be at least 1.")
    if minimum_amp_scale_floor <= 0:
        raise ValueError("minimum_amp_scale_floor must be positive.")
    if ignore_index is None:
        ignore_index = int(getattr(loss_fn, "ignore_index", 255))
    else:
        ignore_index = int(ignore_index)

    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)
    if total_batches == 0:
        raise ValueError("Training dataloader contains no batches.")
    rolling_loss: Optional[float] = None
    samples_seen = 0
    optimizer_steps = 0
    processed_batches = 0
    skipped_all_ignore_batches = 0
    skipped_amp_overflow_steps = 0
    consecutive_amp_overflow_steps = 0
    pending_supervised_batches = 0
    accumulation_sample_ids: list[str] = []
    minimum_amp_scale: Optional[float] = float(scaler.get_scale()) if amp_enabled and scaler is not None else None
    started_at = perf_counter()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader, start=1):
        images, masks, sample_ids = _unpack_training_batch(batch)
        images = images.to(device)
        masks = masks.to(device)
        batch_size = int(images.shape[0])

        valid_pixel_count = int((masks != ignore_index).sum().item())
        if valid_pixel_count == 0:
            skipped_all_ignore_batches += 1
            _emit_training_diagnostic(
                run_logger,
                event_type="all_ignore_batch",
                payload={
                    "epoch": epoch,
                    "batch": batch_idx,
                    "batch_size": batch_size,
                    "sample_ids": sample_ids,
                    "valid_pixel_count": 0,
                },
            )
            continue

        unique_target_ids = sorted(int(value) for value in torch.unique(masks).tolist())
        images_finite = bool(torch.isfinite(images).all().item())
        if not images_finite:
            diagnostic = {
                "epoch": epoch,
                "batch": batch_idx,
                "sample_ids": sample_ids,
                "valid_pixel_count": valid_pixel_count,
                "unique_target_ids": unique_target_ids,
                "images_finite": False,
                "logits_finite": False,
                "loss_finite": False,
                "amp_enabled": bool(amp_enabled),
                "grad_scaler_scale": float(scaler.get_scale()) if scaler is not None else None,
            }
            _emit_training_diagnostic(run_logger, event_type="non_finite_inputs", payload=diagnostic)
            raise RuntimeError(
                "Training inputs are non-finite for a batch with valid target pixels "
                f"(epoch={epoch}, batch={batch_idx}, sample_ids={sample_ids})."
            )

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images)
            logits_finite = bool(torch.isfinite(logits).all().item())
            if not logits_finite:
                diagnostic = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "sample_ids": sample_ids,
                    "valid_pixel_count": valid_pixel_count,
                    "unique_target_ids": unique_target_ids,
                    "images_finite": images_finite,
                    "logits_finite": False,
                    "loss_finite": False,
                    "amp_enabled": bool(amp_enabled),
                    "grad_scaler_scale": float(scaler.get_scale()) if scaler is not None else None,
                }
                _emit_training_diagnostic(run_logger, event_type="non_finite_logits", payload=diagnostic)
                raise RuntimeError(
                    "Training logits are non-finite for a batch with valid target pixels "
                    f"(epoch={epoch}, batch={batch_idx}, sample_ids={sample_ids})."
                )
            loss = loss_fn(logits, masks)

        loss_finite = bool(torch.isfinite(loss).item())
        if not loss_finite:
            diagnostic = {
                "epoch": epoch,
                "batch": batch_idx,
                "sample_ids": sample_ids,
                "valid_pixel_count": valid_pixel_count,
                "unique_target_ids": unique_target_ids,
                "images_finite": images_finite,
                "logits_finite": True,
                "loss_finite": False,
                "amp_enabled": bool(amp_enabled),
                "grad_scaler_scale": float(scaler.get_scale()) if scaler is not None else None,
            }
            _emit_training_diagnostic(run_logger, event_type="non_finite_loss", payload=diagnostic)
            raise RuntimeError(
                "Training loss is non-finite for a batch with valid target pixels "
                f"(epoch={epoch}, batch={batch_idx}, sample_ids={sample_ids})."
            )

        scaled_loss = loss / gradient_accumulation_steps
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        pending_supervised_batches += 1
        if sample_ids is not None:
            accumulation_sample_ids.extend(sample_ids)
        if pending_supervised_batches >= gradient_accumulation_steps:
            finalize_result = _finalize_optimizer_step(
                model=model,
                optimizer=optimizer,
                amp_enabled=amp_enabled,
                scaler=scaler,
                run_logger=run_logger,
                epoch=epoch,
                accumulation_end_batch=batch_idx,
                accumulation_sample_ids=accumulation_sample_ids,
                accumulation_supervised_batches=pending_supervised_batches,
            )
            if finalize_result["optimizer_step_taken"]:
                optimizer_steps += 1
                consecutive_amp_overflow_steps = 0
            else:
                skipped_amp_overflow_steps += 1
                consecutive_amp_overflow_steps += 1
                if consecutive_amp_overflow_steps >= max_consecutive_amp_overflow_steps:
                    raise RuntimeError(
                        "AMP overflow skipped too many consecutive optimizer steps; "
                        f"epoch={epoch}, batch={batch_idx}, consecutive_skips={consecutive_amp_overflow_steps}."
                    )
                if finalize_result["new_scale"] is not None and float(finalize_result["new_scale"]) <= minimum_amp_scale_floor:
                    raise RuntimeError(
                        "AMP GradScaler scale reached configured floor after overflow; "
                        f"epoch={epoch}, batch={batch_idx}, scale={float(finalize_result[new_scale])}."
                    )
            if minimum_amp_scale is not None and finalize_result["new_scale"] is not None:
                minimum_amp_scale = min(minimum_amp_scale, float(finalize_result["new_scale"]))
            pending_supervised_batches = 0
            accumulation_sample_ids = []

        loss_value = float(loss.item())
        total_loss += loss_value
        rolling_loss = loss_value if rolling_loss is None else 0.9 * rolling_loss + 0.1 * loss_value
        samples_seen += batch_size
        processed_batches += 1

        elapsed_seconds = perf_counter() - started_at
        throughput = samples_seen / elapsed_seconds if elapsed_seconds > 0 else None
        eta_seconds = (elapsed_seconds / batch_idx) * (total_batches - batch_idx)
        if run_logger is not None and (batch_idx % batch_log_interval == 0 or batch_idx == total_batches):
            run_logger.log_batch(
                epoch=epoch,
                batch=batch_idx,
                total_batches=total_batches,
                loss=loss_value,
                smoothed_loss=rolling_loss,
                throughput_samples_per_second=throughput,
                eta_seconds=eta_seconds,
                **_cuda_allocator_metrics(device),
            )

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{total_batches}  loss={loss_value:.4f}")

    if pending_supervised_batches > 0:
        finalize_result = _finalize_optimizer_step(
            model=model,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
            scaler=scaler,
            run_logger=run_logger,
            epoch=epoch,
            accumulation_end_batch=total_batches,
            accumulation_sample_ids=accumulation_sample_ids,
            accumulation_supervised_batches=pending_supervised_batches,
        )
        if finalize_result["optimizer_step_taken"]:
            optimizer_steps += 1
        else:
            skipped_amp_overflow_steps += 1
            if finalize_result["new_scale"] is not None and float(finalize_result["new_scale"]) <= minimum_amp_scale_floor:
                raise RuntimeError(
                    "AMP GradScaler scale reached configured floor after overflow at epoch end; "
                    f"epoch={epoch}, scale={float(finalize_result[new_scale])}."
                )
        if minimum_amp_scale is not None and finalize_result["new_scale"] is not None:
            minimum_amp_scale = min(minimum_amp_scale, float(finalize_result["new_scale"]))

    if processed_batches == 0:
        raise RuntimeError(
            "Training epoch contains no supervised batches with valid target pixels; "
            "all batches were all-ignore and skipped."
        )

    return {
        "mean_loss": total_loss / processed_batches,
        "optimizer_steps": optimizer_steps,
        "processed_batches": processed_batches,
        "skipped_all_ignore_batches": skipped_all_ignore_batches,
        "skipped_amp_overflow_steps": skipped_amp_overflow_steps,
        "minimum_amp_scale": minimum_amp_scale,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int = 4,
    ignore_index: int = 255,
    return_per_class_iou: bool = False,
    return_detailed_metrics: bool = False,
    *,
    epoch: Optional[int] = None,
    train_loss: Optional[float] = None,
    learning_rate: Optional[float] = None,
    epoch_duration_seconds: Optional[float] = None,
    run_logger: Optional[Any] = None,
    unweighted_loss_fn: Optional[nn.Module] = None,
) -> dict:
    """Evaluate the model on a validation or test DataLoader.

    Parameters
    ----------
    model : nn.Module
        Segmentation model (same architecture as used during training).
    dataloader : DataLoader
        Validation / test data loader.
    loss_fn : nn.Module
        Loss function used to compute validation loss.
    device : torch.device
        Device to run computations on.
    num_classes : int
        Number of semantic classes (not counting the ignore class).
    ignore_index : int
        Pixels with this label are excluded from metric computation.
    return_per_class_iou : bool
        If True, also compute and return per-class IoU in key
        ``"per_class_iou"``.

    Returns
    -------
    dict
        Keys: ``"val_loss"`` (float), ``"pixel_accuracy"`` (float),
        ``"mean_iou"`` (float), ``"finite_loss_batches"`` (int), and
        ``"skipped_all_ignore_loss_batches"`` (int).
    """
    if run_logger is not None and epoch is None:
        raise ValueError("epoch is required when run_logger is provided.")

    model.eval()
    total_loss = 0.0
    total_unweighted_loss = 0.0
    finite_loss_batches = 0
    skipped_all_ignore_loss_batches = 0
    total_correct = 0
    total_valid = 0
    class_intersections = torch.zeros(num_classes, dtype=torch.long)
    class_unions = torch.zeros(num_classes, dtype=torch.long)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)              # [B, num_classes, H, W]
            valid = masks != ignore_index
            if valid.any():
                loss = loss_fn(logits, masks)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Evaluation loss is non-finite for a batch containing valid target pixels."
                    )
                total_loss += loss.item()
                if unweighted_loss_fn is not None:
                    unweighted_loss = unweighted_loss_fn(logits, masks)
                    if not torch.isfinite(unweighted_loss):
                        raise RuntimeError("Unweighted evaluation loss is non-finite for valid target pixels.")
                    total_unweighted_loss += unweighted_loss.item()
                finite_loss_batches += 1
            else:
                skipped_all_ignore_loss_batches += 1

            # Convert logits to predicted class IDs
            preds = logits.argmax(dim=1)        # [B, H, W]

            confusion_matrix += segmentation_confusion_matrix(
                preds.detach().cpu(),
                masks.detach().cpu(),
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

            total_correct += ((preds == masks) & valid).sum().item()
            total_valid += valid.sum().item()

            for class_idx in range(num_classes):
                pred_c = (preds == class_idx) & valid
                true_c = (masks == class_idx) & valid
                class_intersections[class_idx] += (pred_c & true_c).sum().cpu()
                class_unions[class_idx] += (pred_c | true_c).sum().cpu()

    acc = total_correct / max(total_valid, 1)

    per_class_iou = []
    for class_idx in range(num_classes):
        union = int(class_unions[class_idx].item())
        if union == 0:
            per_class_iou.append(None)
        else:
            intersection = int(class_intersections[class_idx].item())
            per_class_iou.append(intersection / union)

    valid_scores = [score for score in per_class_iou if score is not None]
    miou = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0

    if finite_loss_batches == 0:
        raise RuntimeError("Evaluation split contains no batches with valid target pixels.")

    results = {
        "val_loss": total_loss / finite_loss_batches,
        "pixel_accuracy": acc,
        "mean_iou": miou,
        "finite_loss_batches": finite_loss_batches,
        "skipped_all_ignore_loss_batches": skipped_all_ignore_loss_batches,
    }
    if unweighted_loss_fn is not None:
        results["unweighted_val_loss"] = total_unweighted_loss / finite_loss_batches

    if return_per_class_iou:
        results["per_class_iou"] = per_class_iou

    detailed_metrics = segmentation_metrics_from_confusion_matrix(confusion_matrix)
    if return_detailed_metrics:
        results.update(detailed_metrics)

    if run_logger is not None:
        from src.research_console.schema import ClassMetrics, EpochMetrics

        class_names = ("soil", "bedrock", "sand", "big_rock")
        per_class = {
            class_names[index] if index < len(class_names) else f"class_{index}": ClassMetrics(
                **{key: value for key, value in metrics.items() if key != "class_index"}
            )
            for index, metrics in enumerate(detailed_metrics["per_class"])
        }
        run_logger.log_epoch(
            EpochMetrics(
                timestamp=datetime.now(timezone.utc),
                epoch=epoch,
                train_loss=train_loss,
                val_loss=results["val_loss"],
                pixel_accuracy=results["pixel_accuracy"],
                mean_iou=results["mean_iou"],
                per_class=per_class,
                learning_rate=learning_rate,
                epoch_duration_seconds=epoch_duration_seconds,
                confusion_matrix=detailed_metrics["confusion_matrix"],
            )
        )

    return results
