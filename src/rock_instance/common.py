"""Shared contracts for development-only rock-instance tooling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.paper_reproduction import validate_reproduction_manifest


DEVELOPMENT_SPLITS = frozenset({"train", "val"})


def require_development_splits(splits: Iterable[str]) -> tuple[str, ...]:
    """Validate and deterministically order the only permissible Sprint 0 splits."""
    requested = tuple(sorted(set(splits)))
    if not requested:
        raise ValueError("At least one development split is required.")
    invalid = sorted(set(requested) - DEVELOPMENT_SPLITS)
    if invalid:
        raise ValueError(
            "Sprint 0 tools may use only train and val manifests; "
            f"received forbidden splits: {invalid}"
        )
    return requested


def load_development_manifest_rows(manifest_root: Path, split_paths: dict[str, str], splits: Iterable[str]) -> list[dict[str, str]]:
    """Load validated train/val rows, annotating each with its source split."""
    rows: list[dict[str, str]] = []
    for split_name in require_development_splits(splits):
        try:
            relative_path = split_paths[split_name]
        except KeyError as error:
            raise ValueError(f"No manifest path was supplied for development split {split_name!r}.") from error
        for row in validate_reproduction_manifest(Path(manifest_root) / relative_path, split_name=split_name):
            rows.append({**row, "split": split_name})
    return sorted(rows, key=lambda row: (row["split"], row["sequence_id"], row["stable_source_image_id"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a stable UTF-8 CSV schema, including a header for zero rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")