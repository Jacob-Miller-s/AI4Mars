"""Centralized runtime paths for local development and Kaggle execution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.data_paths import PROJECT_ROOT


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
DATASET_DIRECTORY_NAME = "ai4mars-dataset-merged-0.6"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


def is_kaggle_runtime(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the process is executing inside a Kaggle notebook runtime."""
    environment = os.environ if environ is None else environ
    return bool(environment.get("KAGGLE_KERNEL_RUN_TYPE") or environment.get("KAGGLE_URL_BASE"))


def is_kaggle_input_path(path: Path) -> bool:
    """Return whether *path* is inside Kaggle's read-only input mount."""
    normalized = Path(path).as_posix().replace("\\", "/").rstrip("/").lower()
    return normalized == "/kaggle/input" or "/kaggle/input/" in normalized


def require_writable_path(path: Path) -> Path:
    """Reject paths under Kaggle inputs and return a normalized writable path."""
    resolved = Path(path).expanduser()
    if is_kaggle_input_path(resolved):
        raise ValueError(f"Generated files cannot be written under Kaggle inputs: {resolved}")
    return resolved


def reject_local_kaggle_mount(path: Path | str, *, kaggle: bool) -> None:
    """Fail fast when a Kaggle-only path is passed to native local Python.

    Git Bash rewrites ``/kaggle/...`` to ``C:/Program Files/Git/kaggle/...``
    before invoking Python on Windows. That path must never be created locally.
    """
    normalized = str(path).replace("\\", "/").lower()
    kaggle_mount = normalized.startswith("/kaggle/") or "/program files/git/kaggle/" in normalized
    if kaggle_mount and not kaggle:
        raise ValueError(
            "Kaggle paths are valid only inside a Kaggle runtime. Git Bash converted "
            f"the supplied path to {path!s}. Run this command in Kaggle, not Windows."
        )


def locate_dataset_root(candidate_root: Path) -> Path:
    """Resolve an extracted AI4Mars root from a configured dataset mount."""
    candidate_root = Path(candidate_root).expanduser()
    if candidate_root.name == DATASET_DIRECTORY_NAME:
        return candidate_root

    direct = candidate_root / DATASET_DIRECTORY_NAME
    if direct.is_dir():
        return direct

    matches = sorted(candidate_root.glob(f"*/{DATASET_DIRECTORY_NAME}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "More than one AI4Mars dataset root was found. Set --dataset-root to the extracted "
            f"{DATASET_DIRECTORY_NAME} directory explicitly."
        )
    raise FileNotFoundError(
        f"Could not find {DATASET_DIRECTORY_NAME} below configured dataset root: {candidate_root}"
    )


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved filesystem contract for one training invocation."""

    project_root: Path
    dataset_root: Path
    manifest_root: Path
    output_root: Path
    checkpoint_root: Path
    event_root: Path
    cache_root: Path
    run_id: str
    accelerator: str
    kaggle: bool

    def ensure_writable_roots(self) -> None:
        """Create only generated-data directories after rejecting read-only inputs."""
        for path in (self.output_root, self.checkpoint_root, self.event_root, self.cache_root):
            require_writable_path(path).mkdir(parents=True, exist_ok=True)


def resolve_runtime_paths(
    *,
    project_root: Path = PROJECT_ROOT,
    dataset_root: Path | str | None = None,
    manifest_root: Path | str | None = None,
    output_root: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
    event_root: Path | str | None = None,
    cache_root: Path | str | None = None,
    run_id: str | None = None,
    accelerator: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimePaths:
    """Resolve paths with explicit arguments, environment, runtime, then repo defaults.

    The configured dataset root may be either the extracted dataset directory or a
    directory that contains exactly one extracted dataset directory.
    """
    environment = os.environ if environ is None else environ
    project_root = Path(project_root).resolve()
    kaggle = is_kaggle_runtime(environment)

    configured_dataset_root = dataset_root or environment.get("AI4MARS_DATASET_ROOT")
    if configured_dataset_root is None:
        configured_dataset_root = KAGGLE_INPUT_ROOT if kaggle else project_root / "data" / "raw"
    dataset_root_path = locate_dataset_root(Path(configured_dataset_root))

    resolved_manifest_root = Path(
        manifest_root or environment.get("AI4MARS_MANIFEST_ROOT") or project_root / "artifacts" / "manifests"
    )
    default_output_root = KAGGLE_WORKING_ROOT / "ai4mars" if kaggle else project_root / "outputs"
    resolved_output_root = require_writable_path(
        Path(output_root or environment.get("AI4MARS_OUTPUT_ROOT") or default_output_root)
    )
    resolved_event_root = require_writable_path(
        Path(event_root or environment.get("AI4MARS_EVENT_ROOT") or resolved_output_root / "runs")
    )
    resolved_checkpoint_root = require_writable_path(
        Path(checkpoint_root or environment.get("AI4MARS_CHECKPOINT_ROOT") or resolved_event_root / "checkpoints")
    )
    default_cache_root = KAGGLE_WORKING_ROOT / "ai4mars-cache" if kaggle else project_root / "data" / "processed" / "cache"
    resolved_cache_root = require_writable_path(
        Path(cache_root or environment.get("AI4MARS_CACHE_ROOT") or default_cache_root)
    )
    for path in (resolved_output_root, resolved_event_root, resolved_checkpoint_root, resolved_cache_root):
        reject_local_kaggle_mount(path, kaggle=kaggle)
    resolved_run_id = run_id or environment.get("AI4MARS_RUN_ID") or "ai4mars-baseline"
    if not RUN_ID_PATTERN.fullmatch(resolved_run_id):
        raise ValueError(f"Invalid run id: {resolved_run_id!r}")
    resolved_accelerator = (accelerator or environment.get("AI4MARS_ACCELERATOR") or "auto").lower()
    if resolved_accelerator not in {"auto", "cuda", "cpu"}:
        raise ValueError("accelerator must be one of: auto, cuda, cpu")

    return RuntimePaths(
        project_root=project_root,
        dataset_root=dataset_root_path,
        manifest_root=resolved_manifest_root,
        output_root=resolved_output_root,
        checkpoint_root=resolved_checkpoint_root,
        event_root=resolved_event_root,
        cache_root=resolved_cache_root,
        run_id=resolved_run_id,
        accelerator=resolved_accelerator,
        kaggle=kaggle,
    )