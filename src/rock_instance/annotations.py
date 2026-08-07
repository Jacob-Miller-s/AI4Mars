"""Versioned manual-review annotations for the RGB rock-instance pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.rock_instance.common import DEVELOPMENT_SPLITS, require_development_splits


SCHEMA_VERSION = "rock_instance_pilot_annotation_v1"
REVIEW_VERSION = "pilot_v1"
OBJECT_CLASS_NAME = "rock"
OBJECT_CLASS_ID = 1
ANNOTATION_STATUSES = frozenset(
    {"accepted", "rejected_bedrock", "rejected_noise", "split_required", "merge_required", "uncertain", "deferred"}
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a source candidate manifest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    converted = int(value)
    if converted != value and not (isinstance(value, str) and value.strip() == str(converted)):
        raise ValueError(f"{field_name} must be an integer.")
    return converted


def validate_bbox(bbox: Any, *, image_width: int, image_height: int) -> list[int]:
    """Validate an `[x, y, width, height]` box against image geometry."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height].")
    x, y, width, height = (_as_int(value, field_name="bbox") for value in bbox)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("bbox origin must be non-negative and dimensions must be positive.")
    if x + width > image_width or y + height > image_height:
        raise ValueError("bbox extends beyond image geometry.")
    return [x, y, width, height]


def validate_polygon(polygon: Any, *, image_width: int, image_height: int) -> list[list[float]]:
    """Validate a visible-object polygon in source-image pixel coordinates."""
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("polygon must contain at least three [x, y] points.")
    normalized: list[list[float]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("polygon points must be [x, y].")
        x, y = float(point[0]), float(point[1])
        if not 0 <= x < image_width or not 0 <= y < image_height:
            raise ValueError("polygon point lies outside image geometry.")
        normalized.append([x, y])
    return normalized


def validate_annotation(annotation: dict[str, Any], *, image_id: str, image_width: int, image_height: int) -> dict[str, Any]:
    """Validate one reviewer decision without inventing an object from a candidate."""
    required = {
        "instance_id", "image_id", "sequence_id", "bbox", "annotation_status", "discrete_rock",
        "truncated", "occluded", "uncertain", "reviewer_notes", "review_version",
    }
    missing = sorted(required - set(annotation))
    if missing:
        raise ValueError(f"Annotation is missing required fields: {missing}")
    if annotation["image_id"] != image_id:
        raise ValueError("Annotation image_id does not match its review-state image.")
    if annotation["annotation_status"] not in ANNOTATION_STATUSES:
        raise ValueError(f"Unknown annotation_status: {annotation['annotation_status']!r}")
    if annotation["review_version"] != REVIEW_VERSION:
        raise ValueError(f"review_version must be {REVIEW_VERSION!r}.")
    if not isinstance(annotation["instance_id"], str) or not annotation["instance_id"]:
        raise ValueError("instance_id must be a non-empty string.")
    if not isinstance(annotation["discrete_rock"], bool):
        raise ValueError("discrete_rock must be boolean.")
    for name in ("truncated", "occluded", "uncertain"):
        if not isinstance(annotation[name], bool):
            raise ValueError(f"{name} must be boolean.")
    if not isinstance(annotation["reviewer_notes"], str):
        raise ValueError("reviewer_notes must be a string.")
    normalized = dict(annotation)
    normalized["bbox"] = validate_bbox(annotation["bbox"], image_width=image_width, image_height=image_height)
    if "source_candidate_component_id" in annotation and annotation["source_candidate_component_id"] is not None:
        component_id = _as_int(annotation["source_candidate_component_id"], field_name="source_candidate_component_id")
        if component_id < 1:
            raise ValueError("source_candidate_component_id must be positive when supplied.")
        normalized["source_candidate_component_id"] = component_id
    if annotation["annotation_status"] == "accepted":
        if not annotation["discrete_rock"]:
            raise ValueError("Accepted annotations must set discrete_rock=true.")
        normalized["polygon"] = validate_polygon(
            annotation.get("polygon"), image_width=image_width, image_height=image_height
        )
    elif annotation["discrete_rock"]:
        raise ValueError("Only accepted annotations may set discrete_rock=true.")
    if annotation["annotation_status"] == "uncertain" and not annotation["uncertain"]:
        raise ValueError("Uncertain annotations must set uncertain=true.")
    return normalized


def initialize_review_state(candidate_manifest: Path, dataset_root: Path, *, pilot_id: str = "rock_instance_pilot_v0_candidates") -> dict[str, Any]:
    """Create an empty, source-referenced review state from train/val candidates."""
    with Path(candidate_manifest).open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    if not candidates:
        raise ValueError("Candidate manifest contains no rows.")
    image_ids: set[str] = set()
    images: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda row: int(row["pilot_rank"])):
        image_id = candidate["stable_source_image_id"]
        if image_id in image_ids:
            raise ValueError(f"Candidate manifest duplicates image_id={image_id!r}.")
        if candidate.get("split") not in DEVELOPMENT_SPLITS:
            raise ValueError(f"Candidate manifest contains forbidden split: {candidate.get('split')!r}")
        image_path = Path(dataset_root) / candidate["image_path"]
        mask_path = Path(dataset_root) / candidate["mask_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Candidate source image is missing: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Candidate source mask is missing: {mask_path}")
        with Image.open(image_path) as image_file:
            image_width, image_height = image_file.size
        with Image.open(mask_path) as mask_file:
            if mask_file.size != (image_width, image_height):
                raise ValueError(f"Candidate image/mask geometry mismatch: {image_id}")
        image_ids.add(image_id)
        images[image_id] = {
            "image_id": image_id,
            "pilot_rank": _as_int(candidate["pilot_rank"], field_name="pilot_rank"),
            "split": candidate["split"],
            "sequence_id": candidate["sequence_id"],
            "image_path": candidate["image_path"],
            "mask_path": candidate["mask_path"],
            "image_width": image_width,
            "image_height": image_height,
            "selection_strata": candidate["selection_strata"].split("|"),
            "review_status": "unreviewed",
            "reviewer": None,
            "annotations": [],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "pilot_id": pilot_id,
        "review_version": REVIEW_VERSION,
        "object_class": {"id": OBJECT_CLASS_ID, "name": OBJECT_CLASS_NAME},
        "source_candidate_manifest": Path(candidate_manifest).name,
        "source_candidate_manifest_sha256": sha256_file(candidate_manifest),
        "expert_splits_excluded": True,
        "images": images,
    }


def validate_review_state(state: dict[str, Any]) -> None:
    """Reject malformed, expert-contaminated, or internally inconsistent review state."""
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {state.get('schema_version')!r}")
    if state.get("review_version") != REVIEW_VERSION or state.get("expert_splits_excluded") is not True:
        raise ValueError("Review state lacks the required Sprint 0.5 development-only provenance.")
    images = state.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("Review state must contain a non-empty images mapping.")
    instance_ids: set[str] = set()
    for image_id, image in images.items():
        if image.get("image_id") != image_id or image.get("split") not in DEVELOPMENT_SPLITS:
            raise ValueError(f"Invalid development image record: {image_id!r}")
        width, height = image.get("image_width"), image.get("image_height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid image geometry for {image_id!r}")
        for annotation in image.get("annotations", []):
            normalized = validate_annotation(annotation, image_id=image_id, image_width=width, image_height=height)
            if normalized["instance_id"] in instance_ids:
                raise ValueError(f"Duplicate instance_id: {normalized['instance_id']!r}")
            instance_ids.add(normalized["instance_id"])


def load_review_state(path: Path) -> dict[str, Any]:
    """Load and validate resumable review state."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_review_state(state)
    return state


def save_review_state(path: Path, state: dict[str, Any]) -> None:
    """Validate and atomically persist progress after every reviewer action."""
    validate_review_state(state)
    _atomic_write_json(path, state)


def record_annotation(state: dict[str, Any], annotation: dict[str, Any], *, reviewer: str) -> None:
    """Add one human decision and mark its image reviewed or deferred."""
    image_id = annotation.get("image_id")
    image = state.get("images", {}).get(image_id)
    if image is None:
        raise ValueError(f"Unknown review-state image_id: {image_id!r}")
    normalized = validate_annotation(
        annotation,
        image_id=image_id,
        image_width=image["image_width"],
        image_height=image["image_height"],
    )
    existing_ids = {
        item["instance_id"]
        for item in reviewed_annotations(state)
    }
    if normalized["instance_id"] in existing_ids:
        raise ValueError(f"Duplicate instance_id: {normalized['instance_id']!r}")
    image["annotations"].append(normalized)
    image["annotations"].sort(key=lambda item: item["instance_id"])
    image["reviewer"] = reviewer
    image["review_status"] = "deferred" if normalized["annotation_status"] == "deferred" else "reviewed"


def reviewed_annotations(state: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Iterate over annotations in deterministic source-image order."""
    for image_id in sorted(state["images"]):
        yield from state["images"][image_id]["annotations"]


def polygon_to_mask(polygon: list[list[float]], *, image_width: int, image_height: int) -> torch.Tensor:
    """Rasterize a reviewed visible-object polygon to a boolean source-resolution mask."""
    canvas = Image.new("1", (image_width, image_height), color=0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=1)
    return torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).to(dtype=torch.bool)


def maskrcnn_target_for_image(state: dict[str, Any], image_id: str, *, numeric_image_id: int) -> dict[str, torch.Tensor]:
    """Format accepted reviewed polygons as a future torchvision Mask R-CNN target."""
    validate_review_state(state)
    image = state["images"].get(image_id)
    if image is None:
        raise ValueError(f"Unknown image_id: {image_id!r}")
    accepted = [item for item in image["annotations"] if item["annotation_status"] == "accepted"]
    masks = [
        polygon_to_mask(item["polygon"], image_width=image["image_width"], image_height=image["image_height"])
        for item in accepted
    ]
    boxes = torch.tensor(
        [[item["bbox"][0], item["bbox"][1], item["bbox"][0] + item["bbox"][2], item["bbox"][1] + item["bbox"][3]] for item in accepted],
        dtype=torch.float32,
    ).reshape(-1, 4)
    mask_tensor = torch.stack(masks) if masks else torch.zeros((0, image["image_height"], image["image_width"]), dtype=torch.bool)
    return {
        "boxes": boxes,
        "labels": torch.full((len(accepted),), OBJECT_CLASS_ID, dtype=torch.int64),
        "masks": mask_tensor,
        "image_id": torch.tensor([numeric_image_id], dtype=torch.int64),
        "area": mask_tensor.flatten(1).sum(dim=1).to(dtype=torch.float32),
        "iscrowd": torch.zeros((len(accepted),), dtype=torch.int64),
    }