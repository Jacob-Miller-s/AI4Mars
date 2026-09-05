"""Paper-aligned AI4Mars reproduction contracts and audit helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from ai4mars.dataset import normalize_ai4mars_mask


CLASS_NAMES = ("soil", "bedrock", "sand", "big_rock")
CLASS_IDS = frozenset(range(len(CLASS_NAMES)))
IGNORE_INDEX = 255
MSL_NAVCAM_SCOPE = {"mission": "msl", "rover": "curiosity", "camera": "ncam", "label_scheme": "NAV"}
EXPERT_AGREEMENTS = ("min1-100agree", "min2-100agree", "min3-100agree")
EXPERT_SPLIT_AGREEMENTS = {
    "expert_min1": "min1-100agree",
    "expert_min2": "min2-100agree",
    "expert_min3": "min3-100agree",
}


@dataclass(frozen=True)
class ClassComposition:
    manifest_sha256: str
    pixel_counts: tuple[int, ...]
    class_proportions: tuple[float, ...]
    class_weights: tuple[float, ...]
    ignore_pixel_count: int
    valid_pixel_count: int


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    return rows


def manifest_sha256(manifest_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(manifest_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_relative_dataset_path(value: str, field_name: str, row_number: int) -> None:
    if not value or "\\" in value:
        raise ValueError(f"Row {row_number} has a non-portable {field_name}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in path.parts[0] or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Row {row_number} has an unsafe {field_name}: {value!r}")


def _expected_role(split_name: str) -> tuple[str, str]:
    if split_name in {"train", "val"}:
        return "crowdsourced_train", ""
    agreement = EXPERT_SPLIT_AGREEMENTS.get(split_name)
    if agreement is None:
        raise ValueError(f"Unknown expert agreement split: {split_name}")
    return "expert_gold_test", agreement


def validate_reproduction_manifest(
    manifest_path: Path,
    *,
    split_name: str,
    require_deterministic_ordering: bool = True,
) -> list[dict[str, str]]:
    """Validate a paper-reproduction split without accessing the dataset files."""
    expected_role, expected_agreement = _expected_role(split_name)
    rows = read_manifest_rows(manifest_path)
    previous_key: tuple[str, str, str] | None = None
    source_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        for field_name, expected_value in MSL_NAVCAM_SCOPE.items():
            if row.get(field_name) != expected_value:
                raise ValueError(
                    f"Row {row_number} is outside MSL NavCam NAV scope: "
                    f"{field_name}={row.get(field_name)!r}"
                )
        if row.get("label_role") != expected_role:
            raise ValueError(f"Row {row_number} has label_role={row.get('label_role')!r}, expected {expected_role!r}")
        if row.get("agreement_threshold", "") != expected_agreement:
            raise ValueError(
                f"Row {row_number} has agreement_threshold={row.get('agreement_threshold')!r}, "
                f"expected {expected_agreement!r}"
            )
        for field_name in ("dataset_relative_image_path", "dataset_relative_mask_path"):
            _require_relative_dataset_path(row.get(field_name, ""), field_name, row_number)
        source_id = row.get("stable_source_image_id", "")
        sequence_id = row.get("sequence_id", "")
        if not source_id or not sequence_id:
            raise ValueError(f"Row {row_number} must identify its source image and acquisition sequence.")
        if source_id in source_ids:
            raise ValueError(f"Row {row_number} duplicates stable_source_image_id={source_id!r}.")
        source_ids.add(source_id)
        try:
            image_size = (int(row["image_width"]), int(row["image_height"]))
            mask_size = (int(row["mask_width"]), int(row["mask_height"]))
        except (KeyError, ValueError) as error:
            raise ValueError(f"Row {row_number} has invalid geometry metadata.") from error
        if min(*image_size, *mask_size) <= 0 or image_size != mask_size:
            raise ValueError(f"Row {row_number} has mismatched image/mask geometry: {image_size} vs {mask_size}")
        ordering_key = (sequence_id, source_id, row["dataset_relative_mask_path"])
        if require_deterministic_ordering and previous_key is not None and ordering_key < previous_key:
            raise ValueError(f"Manifest ordering is not deterministic at row {row_number}.")
        previous_key = ordering_key

    return rows


def validate_manifest_files(manifest_path: Path, dataset_root: Path, *, split_name: str) -> None:
    """Validate paths, source geometry, and NAV label IDs against the mounted dataset."""
    for row_number, row in enumerate(validate_reproduction_manifest(manifest_path, split_name=split_name), start=2):
        image_path = Path(dataset_root) / row["dataset_relative_image_path"]
        mask_path = Path(dataset_root) / row["dataset_relative_mask_path"]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Row {row_number} references a missing image or mask.")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f"Row {row_number} image/mask geometry differs on disk: {image.size} vs {mask.size}")
            if image.size != (int(row["image_width"]), int(row["image_height"])):
                raise ValueError(f"Row {row_number} geometry differs from manifest metadata.")
            labels = np.asarray(mask, dtype=np.int64)
        labels = normalize_ai4mars_mask(labels, mask_path)
        observed = set(np.unique(labels).tolist())
        if not observed.issubset(CLASS_IDS | {IGNORE_INDEX}):
            raise ValueError(f"Row {row_number} has invalid NAV label IDs: {sorted(observed)}")


def assert_no_reproduction_leakage(manifests: Mapping[str, Path]) -> None:
    """Reject source-image or acquisition-sequence leakage into expert evaluation."""
    parsed = {name: validate_reproduction_manifest(path, split_name=name) for name, path in manifests.items()}
    identifiers = {
        name: {
            "source": {row["stable_source_image_id"] for row in rows},
            "sequence": {row["sequence_id"] for row in rows},
        }
        for name, rows in parsed.items()
    }
    for left_name, right_name in (("train", "val"),):
        if left_name in identifiers and right_name in identifiers:
            for key in ("source", "sequence"):
                if identifiers[left_name][key] & identifiers[right_name][key]:
                    raise ValueError(f"{key} leakage detected between {left_name} and {right_name}.")
    for expert_name in (name for name in manifests if name.startswith("expert_")):
        for development_name in ("train", "val"):
            if development_name not in identifiers:
                continue
            for key in ("source", "sequence"):
                if identifiers[development_name][key] & identifiers[expert_name][key]:
                    raise ValueError(f"{key} leakage detected between {development_name} and {expert_name}.")


def summarize_reproduction_manifests(manifests: Mapping[str, Path]) -> dict[str, dict[str, int]]:
    """Return deterministic counts by scope, split role, and agreement level."""
    summary: dict[str, Counter[str]] = {}
    for split_name, path in sorted(manifests.items()):
        rows = validate_reproduction_manifest(path, split_name=split_name)
        counts: Counter[str] = Counter()
        for row in rows:
            counts[f"mission:{row['mission']}"] += 1
            counts[f"rover:{row['rover']}"] += 1
            counts[f"camera:{row['camera']}"] += 1
            counts[f"label_role:{row['label_role']}"] += 1
            counts[f"agreement:{row.get('agreement_threshold') or 'none'}"] += 1
        summary[split_name] = dict(sorted(counts.items()))
    return summary


def compute_paper_class_composition(train_manifest: Path) -> ClassComposition:
    """Compute the paper rule $w_c = 1 - p_c$ from merged MSL NavCam training rows."""
    rows = validate_reproduction_manifest(train_manifest, split_name="train")
    pixel_counts = [0] * len(CLASS_NAMES)
    ignore_pixel_count = 0
    for row_number, row in enumerate(rows, start=2):
        try:
            counts = json.loads(row["per_class_pixel_counts_json"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Row {row_number} has invalid per-class pixel counts.") from error
        for label_text, count in counts.items():
            label = int(label_text)
            value = int(count)
            if value < 0:
                raise ValueError(f"Row {row_number} has a negative pixel count.")
            if label in CLASS_IDS:
                pixel_counts[label] += value
            elif label == IGNORE_INDEX:
                ignore_pixel_count += value
            else:
                raise ValueError(f"Row {row_number} has invalid NAV label ID {label}.")
    valid_pixel_count = sum(pixel_counts)
    if valid_pixel_count == 0:
        raise ValueError("Training manifest contains no labeled NAV pixels.")
    proportions = tuple(count / valid_pixel_count for count in pixel_counts)
    return ClassComposition(
        manifest_sha256=manifest_sha256(train_manifest),
        pixel_counts=tuple(pixel_counts),
        class_proportions=proportions,
        class_weights=tuple(1.0 - value for value in proportions),
        ignore_pixel_count=ignore_pixel_count,
        valid_pixel_count=valid_pixel_count,
    )