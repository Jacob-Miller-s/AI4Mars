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
PROTOCOL_V1_INITIAL = "v1.0-initial"
PROTOCOL_V2_CALIBRATION_RESOLVED = "v2.0-calibration-resolved"
ANNOTATION_STATUSES = frozenset(
    {"accepted", "rejected_bedrock", "rejected_noise", "split_required", "merge_required", "uncertain", "deferred"}
)
IMAGE_REVIEW_STATUSES = frozenset({"unreviewed", "in_progress", "reviewed", "deferred"})
TERMINAL_ANNOTATION_STATUSES = frozenset({"accepted", "rejected_bedrock", "rejected_noise", "uncertain"})
RESOLUTION_TYPES = frozenset({"split", "merge"})


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


def _normalize_component_ids(component_ids: Any, *, field_name: str) -> list[int]:
    if not isinstance(component_ids, (list, tuple)) or not component_ids:
        raise ValueError(f"{field_name} must contain at least one component ID.")
    normalized = [_as_int(component_id, field_name=field_name) for component_id in component_ids]
    if any(component_id < 1 for component_id in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique positive component IDs.")
    return sorted(normalized)


def annotation_component_ids(annotation: dict[str, Any]) -> list[int]:
    """Return canonical plural source-component provenance for an annotation."""
    component_ids = annotation.get("source_candidate_component_ids")
    if component_ids is not None:
        return _normalize_component_ids(component_ids, field_name="source_candidate_component_ids")
    component_id = annotation.get("source_candidate_component_id")
    if component_id is None:
        return []
    return [_as_int(component_id, field_name="source_candidate_component_id")]


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
    if "source_candidate_component_ids" in annotation:
        component_ids = _normalize_component_ids(
            annotation["source_candidate_component_ids"], field_name="source_candidate_component_ids"
        )
        scalar_component_id = normalized.get("source_candidate_component_id")
        if scalar_component_id is not None and scalar_component_id not in component_ids:
            raise ValueError("source_candidate_component_id must be included in source_candidate_component_ids.")
        normalized["source_candidate_component_ids"] = component_ids
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


def configure_review_scope(
    state: dict[str, Any],
    *,
    name: str,
    image_ids: Iterable[str],
    source_manifest: Path,
) -> None:
    """Persist a bounded review queue without changing any human decisions."""
    if any(image["annotations"] for image in state.get("images", {}).values()):
        raise ValueError("A review scope cannot be changed after annotation decisions exist.")
    ordered_ids = list(image_ids)
    if not isinstance(name, str) or not name:
        raise ValueError("Review scope name must be a non-empty string.")
    if not ordered_ids or any(not isinstance(image_id, str) or not image_id for image_id in ordered_ids):
        raise ValueError("Review scope must contain at least one non-empty image ID.")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Review scope contains duplicate image IDs.")
    unknown_ids = sorted(set(ordered_ids) - set(state.get("images", {})))
    if unknown_ids:
        raise ValueError(f"Review scope contains images absent from review state: {unknown_ids}")
    manifest_path = Path(source_manifest)
    state["review_scope"] = {
        "name": name,
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": sha256_file(manifest_path),
        "image_ids": ordered_ids,
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
    review_scope = state.get("review_scope")
    if review_scope is not None:
        required_scope_fields = {"name", "source_manifest", "source_manifest_sha256", "image_ids"}
        if not isinstance(review_scope, dict) or required_scope_fields - set(review_scope):
            raise ValueError("Review scope is missing required provenance fields.")
        scope_image_ids = review_scope["image_ids"]
        if (
            not isinstance(review_scope["name"], str)
            or not review_scope["name"]
            or not isinstance(review_scope["source_manifest"], str)
            or not review_scope["source_manifest"]
            or not isinstance(review_scope["source_manifest_sha256"], str)
            or len(review_scope["source_manifest_sha256"]) != 64
            or not isinstance(scope_image_ids, list)
            or not scope_image_ids
            or any(not isinstance(image_id, str) or not image_id for image_id in scope_image_ids)
            or len(set(scope_image_ids)) != len(scope_image_ids)
        ):
            raise ValueError("Review scope is invalid.")
        unknown_scope_ids = sorted(set(scope_image_ids) - set(images))
        if unknown_scope_ids:
            raise ValueError(f"Review scope contains unknown image IDs: {unknown_scope_ids}")
    component_review = state.get("component_review")
    if component_review is not None:
        required_component_review_fields = {"strict_completion", "component_manifest", "component_manifest_sha256"}
        if (
            not isinstance(component_review, dict)
            or required_component_review_fields - set(component_review)
            or component_review["strict_completion"] is not True
            or not isinstance(component_review["component_manifest"], str)
            or not component_review["component_manifest"]
            or not isinstance(component_review["component_manifest_sha256"], str)
            or len(component_review["component_manifest_sha256"]) != 64
        ):
            raise ValueError("Corrected calibration component-review provenance is invalid.")
        if review_scope is None or review_scope["name"] not in {"calibration", "calibration_repeat", "calibration_clarification"}:
            raise ValueError("Corrected calibration component review requires a calibration, calibration_repeat, or calibration_clarification scope.")
        protocol = state.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("version") != PROTOCOL_V2_CALIBRATION_RESOLVED
            or not isinstance(protocol.get("path"), str)
            or not protocol["path"]
            or not isinstance(protocol.get("sha256"), str)
            or len(protocol["sha256"]) != 64
        ):
            raise ValueError("Corrected calibration state requires v2.0 protocol provenance.")
        initial_reference = state.get("initial_calibration_reference")
        if (
            not isinstance(initial_reference, dict)
            or not isinstance(initial_reference.get("snapshot_path"), str)
            or not initial_reference["snapshot_path"]
            or not isinstance(initial_reference.get("snapshot_sha256"), str)
            or len(initial_reference["snapshot_sha256"]) != 64
            or not isinstance(initial_reference.get("decisions"), list)
        ):
            raise ValueError("Corrected calibration state requires immutable initial-calibration provenance.")
        resolution_records = state.get("resolution_records")
        if not isinstance(resolution_records, list):
            raise ValueError("Corrected calibration state requires a resolution_records list.")
        resolution_ids = [record.get("resolution_id") for record in resolution_records if isinstance(record, dict)]
        if len(resolution_ids) != len(resolution_records) or len(set(resolution_ids)) != len(resolution_ids):
            raise ValueError("Resolution record IDs must be present and unique.")
    instance_ids: set[str] = set()
    for image_id, image in images.items():
        if image.get("image_id") != image_id or image.get("split") not in DEVELOPMENT_SPLITS:
            raise ValueError(f"Invalid development image record: {image_id!r}")
        if image.get("review_status") not in IMAGE_REVIEW_STATUSES:
            raise ValueError(f"Invalid review status for {image_id!r}")
        width, height = image.get("image_width"), image.get("image_height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid image geometry for {image_id!r}")
        if component_review is not None and image_id in review_scope["image_ids"]:
            candidate_component_ids = _normalize_component_ids(
                image.get("candidate_component_ids"), field_name="candidate_component_ids"
            )
            if (
                not isinstance(image.get("obvious_candidate_independent_rock_observed"), bool)
                or not isinstance(image.get("candidate_independent_observation_note"), str)
            ):
                raise ValueError(f"Invalid candidate-independent observation for {image_id!r}")
        for annotation in image.get("annotations", []):
            normalized = validate_annotation(annotation, image_id=image_id, image_width=width, image_height=height)
            if normalized["instance_id"] in instance_ids:
                raise ValueError(f"Duplicate instance_id: {normalized['instance_id']!r}")
            instance_ids.add(normalized["instance_id"])
            if component_review is not None and image_id in review_scope["image_ids"]:
                if not set(annotation_component_ids(normalized)) <= set(candidate_component_ids):
                    raise ValueError("Corrected calibration annotations must reference known candidate components.")
    if component_review is not None:
        for record in state["resolution_records"]:
            _validate_resolution_record(state, record)
        for image_id in review_scope["image_ids"]:
            image = images[image_id]
            if image["review_status"] == "reviewed":
                missing_component_ids = unresolved_candidate_component_ids(state, image_id)
                if missing_component_ids:
                    raise ValueError(
                        f"Cannot mark {image_id!r} reviewed; unresolved candidate component IDs: {missing_component_ids}"
                    )


def load_review_state(path: Path) -> dict[str, Any]:
    """Load and validate resumable review state."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_review_state(state)
    return state


def save_review_state(path: Path, state: dict[str, Any]) -> None:
    """Validate and atomically persist progress after every reviewer action."""
    validate_review_state(state)
    _atomic_write_json(path, state)


def record_annotation(
    state: dict[str, Any],
    annotation: dict[str, Any],
    *,
    reviewer: str,
    image_review_status: str | None = None,
) -> None:
    """Add one human decision and preserve the image's explicit review progress."""
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
    if image_review_status is not None:
        if image_review_status not in IMAGE_REVIEW_STATUSES - {"unreviewed"}:
            raise ValueError("image_review_status must be in_progress, reviewed, or deferred.")
        image["review_status"] = image_review_status
    else:
        strict_component_review = (
            "component_review" in state
            and image_id in state.get("review_scope", {}).get("image_ids", [])
        )
        image["review_status"] = (
            "deferred"
            if normalized["annotation_status"] == "deferred"
            else "in_progress" if strict_component_review else "reviewed"
        )


def finish_image_review(state: dict[str, Any], image_id: str, *, reviewer: str) -> None:
    """Explicitly mark a human-reviewed image complete after all decisions are saved."""
    image = state.get("images", {}).get(image_id)
    if image is None:
        raise ValueError(f"Unknown review-state image_id: {image_id!r}")
    if not image["annotations"]:
        raise ValueError("Cannot finish an image without at least one human decision.")
    missing_component_ids = unresolved_candidate_component_ids(state, image_id)
    if missing_component_ids:
        raise ValueError(f"Cannot finish image; unresolved candidate component IDs: {missing_component_ids}")
    image["reviewer"] = reviewer
    image["review_status"] = "reviewed"


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
    if any(item["annotation_status"] == "uncertain" for item in image["annotations"]):
        raise ValueError("Images containing terminal uncertain annotations are excluded from ordinary Mask R-CNN targets.")
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


def configure_component_review(
    state: dict[str, Any],
    *,
    component_manifest: Path,
    component_ids_by_image: dict[str, Iterable[int]],
    protocol_path: Path,
    initial_calibration_reference: dict[str, Any],
) -> None:
    """Configure a corrected calibration state with strict candidate-component coverage."""
    if any(image["annotations"] for image in state.get("images", {}).values()):
        raise ValueError("Component review requirements must be configured before human decisions exist.")
    if state.get("review_scope", {}).get("name") != "calibration":
        raise ValueError("Strict component review is only valid for an active calibration scope.")
    scope_image_ids = state["review_scope"]["image_ids"]
    unknown_image_ids = sorted(set(component_ids_by_image) - set(state["images"]))
    if unknown_image_ids:
        raise ValueError(f"Component manifest includes unknown review images: {unknown_image_ids}")
    for image_id in scope_image_ids:
        component_ids = _normalize_component_ids(
            list(component_ids_by_image.get(image_id, [])), field_name="candidate_component_ids"
        )
        state["images"][image_id]["candidate_component_ids"] = component_ids
        state["images"][image_id]["obvious_candidate_independent_rock_observed"] = False
        state["images"][image_id]["candidate_independent_observation_note"] = ""
    state["component_review"] = {
        "strict_completion": True,
        "component_manifest": Path(component_manifest).name,
        "component_manifest_sha256": sha256_file(component_manifest),
    }
    state["protocol"] = {
        "version": PROTOCOL_V2_CALIBRATION_RESOLVED,
        "path": str(Path(protocol_path)),
        "sha256": sha256_file(protocol_path),
    }
    state["initial_calibration_reference"] = initial_calibration_reference
    state["resolution_records"] = []


def _initial_decisions_for_image(state: dict[str, Any], image_id: str) -> set[str]:
    reference = state.get("initial_calibration_reference", {})
    decisions = reference.get("decisions", [])
    return {
        decision["instance_id"]
        for decision in decisions
        if decision.get("image_id") == image_id
    }


def initial_calibration_reference(snapshot_path: Path) -> dict[str, Any]:
    """Extract immutable initial-review IDs and provenance for resolution links."""
    snapshot_path = Path(snapshot_path)
    initial_state = load_review_state(snapshot_path)
    decisions = [
        {"instance_id": annotation["instance_id"], "image_id": annotation["image_id"]}
        for annotation in reviewed_annotations(initial_state)
    ]
    return {
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "protocol_version": PROTOCOL_V1_INITIAL,
        "decisions": decisions,
    }


def _validate_resolution_record(state: dict[str, Any], record: dict[str, Any]) -> None:
    required = {
        "resolution_id", "resolution_type", "image_id", "sequence_id", "source_candidate_component_ids",
        "initial_decision_instance_ids", "resolved_annotation_instance_ids", "reviewer_notes",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Resolution record is missing required fields: {missing}")
    if not isinstance(record["resolution_id"], str) or not record["resolution_id"]:
        raise ValueError("resolution_id must be a non-empty string.")
    if record["resolution_type"] not in RESOLUTION_TYPES:
        raise ValueError("resolution_type must be split or merge.")
    image = state["images"].get(record["image_id"])
    if image is None or record["sequence_id"] != image["sequence_id"]:
        raise ValueError("Resolution record image or sequence provenance is invalid.")
    component_ids = _normalize_component_ids(
        record["source_candidate_component_ids"], field_name="source_candidate_component_ids"
    )
    candidate_component_ids = set(image.get("candidate_component_ids", []))
    if candidate_component_ids and not set(component_ids) <= candidate_component_ids:
        raise ValueError("Resolution record references a component outside its image candidate manifest.")
    initial_ids = record["initial_decision_instance_ids"]
    if not isinstance(initial_ids, list) or not initial_ids or any(not isinstance(item, str) or not item for item in initial_ids):
        raise ValueError("Resolution record must reference at least one initial decision ID.")
    if len(set(initial_ids)) != len(initial_ids) or not set(initial_ids) <= _initial_decisions_for_image(state, record["image_id"]):
        raise ValueError("Resolution record initial decision references are invalid.")
    resolved_ids = record["resolved_annotation_instance_ids"]
    if not isinstance(resolved_ids, list) or not resolved_ids or any(not isinstance(item, str) or not item for item in resolved_ids):
        raise ValueError("Resolution record must reference at least one resolved annotation ID.")
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("Resolution record resolved annotation IDs must be unique.")
    annotations_by_id = {annotation["instance_id"]: annotation for annotation in image["annotations"]}
    if not set(resolved_ids) <= set(annotations_by_id):
        raise ValueError("Resolution record references an unknown resolved annotation.")
    resolved_annotations = [annotations_by_id[instance_id] for instance_id in resolved_ids]
    if any(annotation["annotation_status"] not in TERMINAL_ANNOTATION_STATUSES for annotation in resolved_annotations):
        raise ValueError("Resolution records may reference only terminal annotations.")
    accepted = [annotation for annotation in resolved_annotations if annotation["annotation_status"] == "accepted"]
    if record["resolution_type"] == "merge" and accepted:
        if len(accepted) != 1 or len(resolved_annotations) != 1:
            raise ValueError("A merge resolution must produce exactly one accepted instance.")
        if set(annotation_component_ids(accepted[0])) != set(component_ids):
            raise ValueError("Accepted merge instance must preserve every contributing component ID.")
    if record["resolution_type"] == "split" and accepted:
        if len(component_ids) != 1 or len(accepted) < 2:
            raise ValueError("A split resolution must produce at least two accepted children from one parent component.")
        if any(set(component_ids) - set(annotation_component_ids(annotation)) for annotation in accepted):
            raise ValueError("Every accepted split child must preserve its parent component ID.")
    if not accepted and len(resolved_annotations) != 1:
        raise ValueError("A non-accepted split or merge disposition must use one terminal annotation.")
    if not isinstance(record["reviewer_notes"], str):
        raise ValueError("Resolution reviewer_notes must be a string.")


def record_resolution(state: dict[str, Any], resolution_record: dict[str, Any]) -> None:
    """Append one immutable human split/merge resolution after validating its initial links."""
    if "component_review" not in state:
        raise ValueError("Resolution records are only valid for corrected calibration states.")
    record = dict(resolution_record)
    existing_ids = {item.get("resolution_id") for item in state.get("resolution_records", [])}
    if record.get("resolution_id") in existing_ids:
        raise ValueError(f"Duplicate resolution_id: {record.get('resolution_id')!r}")
    _validate_resolution_record(state, record)
    state["resolution_records"].append(record)
    state["resolution_records"].sort(key=lambda item: item["resolution_id"])


def component_coverage_for_image(state: dict[str, Any], image_id: str) -> set[int]:
    """Return source candidate IDs covered by terminal dispositions or validated resolutions."""
    image = state["images"].get(image_id)
    if image is None:
        raise ValueError(f"Unknown review-state image_id: {image_id!r}")
    covered = {
        component_id
        for annotation in image["annotations"]
        if annotation["annotation_status"] in TERMINAL_ANNOTATION_STATUSES
        for component_id in annotation_component_ids(annotation)
    }
    for record in state.get("resolution_records", []):
        if record["image_id"] == image_id:
            covered.update(record["source_candidate_component_ids"])
    return covered


def unresolved_candidate_component_ids(state: dict[str, Any], image_id: str) -> list[int]:
    """List candidate components not yet covered by a terminal decision or resolution record."""
    image = state["images"].get(image_id)
    if image is None:
        raise ValueError(f"Unknown review-state image_id: {image_id!r}")
    return sorted(set(image.get("candidate_component_ids", [])) - component_coverage_for_image(state, image_id))


def set_candidate_independent_observation(
    state: dict[str, Any], image_id: str, *, observed: bool, note: str,
) -> None:
    """Record an image-level observation without inventing a candidate-independent instance."""
    image = state.get("images", {}).get(image_id)
    if image is None:
        raise ValueError(f"Unknown review-state image_id: {image_id!r}")
    if "candidate_component_ids" not in image:
        raise ValueError("Candidate-independent observations require corrected calibration component review.")
    if not isinstance(observed, bool) or not isinstance(note, str):
        raise ValueError("Candidate-independent observation requires boolean observed and string note values.")
    image["obvious_candidate_independent_rock_observed"] = observed
    image["candidate_independent_observation_note"] = note