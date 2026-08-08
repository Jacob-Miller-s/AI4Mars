"""Prepare and run an isolated visible-extent redraw for fixed accepted rocks."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons
from PIL import Image

from src.dataset import normalize_ai4mars_mask
from src.rock_instance.annotations import (
    REVIEW_VERSION,
    annotation_component_ids,
    sha256_file,
    validate_bbox,
    validate_polygon,
    load_review_state,
    validate_review_state,
)
from src.rock_instance.component_audit import CONNECTIVITY


BOUNDARY_REVIEW_SCHEMA_VERSION = "rock_instance_boundary_review_v1"
BOUNDARY_REVIEW_VERSION = "v2.2-visible-extent-clarified-proposed"
BOUNDARY_SCOPE_NAME = "boundary_clarification"
FINAL_CLARIFICATION_SCHEMA_VERSION = "rock_instance_final_clarification_review_v1"
FINAL_CLARIFICATION_VERSION = "v2.2.1-visible-object-continuity-clarified-proposed"
FINAL_CLARIFICATION_SCOPE_NAME = "final_whole_object_clarification"
FINAL_CLARIFICATION_TARGET_ID = "NLB_483955685EDR_F0470598NCAM00320M1:component-8"
FINAL_CLARIFICATION_PROMPT = (
    "Trace the full defensible visible extent of this already-accepted physical rock. Include all visibly attributable faces "
    "of the same coherent object. Do not trace only a high-contrast face or proposal fragment. Stop at surrounding terrain, "
    "continuous Bedrock, another rock, rover hardware/occlusion, or genuinely indeterminate material. Exclude cast shadow "
    "and do not infer hidden geometry."
)
TARGET_STATUSES = frozenset({"unreviewed", "in_progress", "redrawn", "identity_escalated"})
NAV_CONTEXT_COLORS = np.array([(112, 92, 65), (158, 120, 92), (220, 190, 115), (190, 60, 55), (30, 30, 30)], dtype=np.uint8)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selection_rows(path: Path) -> list[dict[str, str]]:
    required = {"target_id", "stable_source_image_id", "source_candidate_component_id", "v21_instance_id", "boundary_question"}
    rows = _read_csv(path)
    if not rows or required - set(rows[0]):
        raise ValueError("Boundary target manifest has missing required fields.")
    target_ids = [row["target_id"] for row in rows]
    if len(rows) != 3 or len(set(target_ids)) != len(target_ids):
        raise ValueError("Boundary review requires exactly three unique target IDs.")
    return rows


def _final_selection_row(path: Path) -> dict[str, str]:
    required = {"target_id", "stable_source_image_id", "source_candidate_component_id", "v21_instance_id", "boundary_question"}
    rows = _read_csv(path)
    if len(rows) != 1 or required - set(rows[0]) or rows[0]["target_id"] != FINAL_CLARIFICATION_TARGET_ID or rows[0]["boundary_question"] != FINAL_CLARIFICATION_PROMPT:
        raise ValueError(f"Final clarification requires exactly {FINAL_CLARIFICATION_TARGET_ID}.")
    return rows[0]


def _review_config(state: dict[str, Any]) -> dict[str, Any]:
    configurations = {
        BOUNDARY_REVIEW_SCHEMA_VERSION: {"review_version": BOUNDARY_REVIEW_VERSION, "scope_name": BOUNDARY_SCOPE_NAME, "target_count": 3, "requires_v22_provenance": False},
        FINAL_CLARIFICATION_SCHEMA_VERSION: {"review_version": FINAL_CLARIFICATION_VERSION, "scope_name": FINAL_CLARIFICATION_SCOPE_NAME, "target_count": 1, "requires_v22_provenance": True},
    }
    try:
        return configurations[state.get("schema_version")]
    except KeyError as error:
        raise ValueError("Unsupported fixed-identity review schema.") from error


def validate_boundary_review_state(state: dict[str, Any]) -> None:
    """Validate a boundary-only state that cannot alter accepted object identity."""
    configuration = _review_config(state)
    if state.get("review_version") != configuration["review_version"] or state.get("expert_splits_excluded") is not True:
        raise ValueError("Boundary review lacks required proposed-protocol provenance.")
    scope = state.get("review_scope", {})
    if scope.get("name") != configuration["scope_name"] or not isinstance(scope.get("target_ids"), list) or len(scope["target_ids"]) != configuration["target_count"]:
        raise ValueError("Boundary review has an invalid fixed target scope.")
    provenance = state.get("provenance", {})
    required_provenance = {
        "primary_state_sha256", "repeat_state_sha256", "v21_state_sha256", "proposed_protocol_sha256",
        "component_manifest_sha256", "historic_annotations_hidden", "v21_polygons_hidden",
    }
    if configuration["requires_v22_provenance"]:
        required_provenance.update({"source_boundary_state_sha256", "v22_polygons_hidden"})
    boolean_provenance = {"historic_annotations_hidden", "v21_polygons_hidden"}
    if configuration["requires_v22_provenance"]:
        boolean_provenance.add("v22_polygons_hidden")
    if required_provenance - set(provenance) or not all(isinstance(provenance[key], str) and len(provenance[key]) == 64 for key in required_provenance - boolean_provenance):
        raise ValueError("Boundary review provenance is invalid.")
    hidden_fields = {"historic_annotations_hidden", "v21_polygons_hidden"}
    if configuration["requires_v22_provenance"]:
        hidden_fields.add("v22_polygons_hidden")
    if any(provenance[field] is not True for field in hidden_fields):
        raise ValueError("Boundary review must attest that prior polygons are hidden.")
    targets = state.get("targets")
    if not isinstance(targets, list) or len(targets) != configuration["target_count"]:
        raise ValueError("Boundary review has an invalid target count.")
    if [target.get("target_id") for target in targets] != scope["target_ids"]:
        raise ValueError("Boundary targets must match the scoped target IDs in order.")
    for target in targets:
        required_target = {
            "target_id", "image_id", "sequence_id", "image_path", "mask_path", "image_width", "image_height",
            "source_candidate_component_id", "v21_instance_id", "object_identity_fixed", "boundary_question",
            "review_status", "reviewer", "polygon", "bbox", "reviewer_notes", "identity_escalation",
            "identity_escalation_note",
        }
        if required_target - set(target):
            raise ValueError("Boundary target is missing required fields.")
        if target["object_identity_fixed"] != "accepted" or target["review_status"] not in TARGET_STATUSES:
            raise ValueError("Boundary target identity or status is invalid.")
        if not isinstance(target["source_candidate_component_id"], int) or target["source_candidate_component_id"] < 1:
            raise ValueError("Boundary target component provenance is invalid.")
        if not isinstance(target["identity_escalation"], bool) or not isinstance(target["identity_escalation_note"], str):
            raise ValueError("Boundary target escalation fields are invalid.")
        if not isinstance(target["reviewer_notes"], str):
            raise ValueError("Boundary reviewer notes must be a string.")
        if target["review_status"] in {"in_progress", "redrawn"}:
            if target["identity_escalation"] or target["polygon"] is None or target["bbox"] is None:
                raise ValueError("An in-progress or redrawn boundary requires a polygon and cannot alter identity.")
            validate_polygon(target["polygon"], image_width=target["image_width"], image_height=target["image_height"])
            validate_bbox(target["bbox"], image_width=target["image_width"], image_height=target["image_height"])
        elif target["review_status"] == "identity_escalated":
            if not target["identity_escalation"] or not target["identity_escalation_note"].strip() or target["polygon"] is not None:
                raise ValueError("An identity escalation requires a note and cannot contain a redraw polygon.")
        elif target["polygon"] is not None or target["bbox"] is not None or target["identity_escalation"]:
            raise ValueError("An unreviewed boundary target cannot contain redraw or escalation output.")


def load_boundary_review_state(path: Path) -> dict[str, Any]:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_boundary_review_state(state)
    return state


def save_boundary_review_state(path: Path, state: dict[str, Any]) -> None:
    validate_boundary_review_state(state)
    _atomic_write_json(path, state)


def _target_by_id(state: dict[str, Any], target_id: str) -> dict[str, Any]:
    for target in state["targets"]:
        if target["target_id"] == target_id:
            return target
    raise ValueError(f"Unknown boundary target: {target_id}")


def _bbox_from_polygon(polygon: list[list[float]]) -> list[int]:
    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    left, top = int(min(x_values)), int(min(y_values))
    return [left, top, max(1, int(max(x_values)) - left + 1), max(1, int(max(y_values)) - top + 1)]


def record_boundary_redraw(state: dict[str, Any], target_id: str, *, polygon: list[list[float]], reviewer: str, notes: str) -> None:
    """Record a polygon only; the target remains an accepted identity by design."""
    target = _target_by_id(state, target_id)
    if target["review_status"] not in {"unreviewed", "in_progress"}:
        raise ValueError("Boundary target is already complete; create a separate review artifact for another redraw.")
    target["polygon"] = validate_polygon(polygon, image_width=target["image_width"], image_height=target["image_height"])
    target["bbox"] = _bbox_from_polygon(target["polygon"])
    target["review_status"] = "in_progress"
    target["reviewer"] = reviewer
    target["reviewer_notes"] = notes


def record_identity_escalation(state: dict[str, Any], target_id: str, *, reviewer: str, note: str) -> None:
    """Record evidence that the fixed identity reference itself needs human escalation, without relabeling it."""
    target = _target_by_id(state, target_id)
    if target["review_status"] not in {"unreviewed", "in_progress"} or not note.strip():
        raise ValueError("An identity escalation requires an incomplete target and a non-empty note.")
    target.update({
        "review_status": "identity_escalated",
        "reviewer": reviewer,
        "polygon": None,
        "bbox": None,
        "reviewer_notes": "",
        "identity_escalation": True,
        "identity_escalation_note": note,
    })


def finish_boundary_target(state: dict[str, Any], target_id: str, *, reviewer: str) -> None:
    target = _target_by_id(state, target_id)
    if target["review_status"] == "in_progress" and target["polygon"] is not None:
        target["review_status"] = "redrawn"
    if target["review_status"] not in {"redrawn", "identity_escalated"}:
        raise ValueError("Boundary target requires a redraw polygon or an explicit identity escalation before completion.")
    target["reviewer"] = reviewer


def _candidate_rows(component_candidates_csv: Path, image_id: str) -> list[dict[str, str]]:
    return [row for row in _read_csv(component_candidates_csv) if row["stable_source_image_id"] == image_id]


def _target_record(v21: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    image_id = row["stable_source_image_id"]
    component_id = int(row["source_candidate_component_id"])
    image = v21["images"].get(image_id)
    if image is None:
        raise ValueError(f"Boundary target image is absent from v2.1 evidence: {image_id}")
    annotation = next((item for item in image["annotations"] if item["instance_id"] == row["v21_instance_id"]), None)
    if annotation is None or annotation["annotation_status"] != "accepted" or component_id not in annotation_component_ids(annotation):
        raise ValueError(f"Boundary target does not reference an accepted v2.1 source component: {row['target_id']}")
    return {
        "target_id": row["target_id"],
        "image_id": image_id,
        "sequence_id": image["sequence_id"],
        "image_path": image["image_path"],
        "mask_path": image["mask_path"],
        "image_width": image["image_width"],
        "image_height": image["image_height"],
        "source_candidate_component_id": component_id,
        "v21_instance_id": row["v21_instance_id"],
        "object_identity_fixed": "accepted",
        "boundary_question": row["boundary_question"],
        "review_status": "unreviewed",
        "reviewer": None,
        "polygon": None,
        "bbox": None,
        "reviewer_notes": "",
        "identity_escalation": False,
        "identity_escalation_note": "",
    }


def _accepted_annotations_for_component(record: dict[str, Any], component_id: int) -> list[dict[str, Any]]:
    return [
        annotation for annotation in record["annotations"]
        if annotation["annotation_status"] == "accepted" and component_id in annotation_component_ids(annotation)
    ]


def _forensic_filename(target_id: str) -> str:
    return f"{target_id.replace(':', '__')}.png"


def _render_forensic_overlay(
    *, primary: dict[str, Any], repeat: dict[str, Any], v21: dict[str, Any], target: dict[str, Any], dataset_root: Path, output_path: Path,
) -> None:
    image_id, component_id = target["image_id"], target["source_candidate_component_id"]
    image = v21["images"][image_id]
    with Image.open(Path(dataset_root) / image["image_path"]) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    mask_path = Path(dataset_root) / image["mask_path"]
    with Image.open(mask_path) as handle:
        mask = normalize_ai4mars_mask(np.asarray(handle, dtype=np.int64), mask_path)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats((mask == 3).astype(np.uint8), connectivity=CONNECTIVITY)
    if component_id >= component_count:
        raise ValueError(f"Boundary target component is absent from source mask: {target['target_id']}")
    candidate_mask = labels == component_id
    terrain_indices = np.where((mask >= 0) & (mask <= 3), mask, 4)
    records = {"primary": primary["images"][image_id], "repeat": repeat["images"][image_id], "v2.1": v21["images"][image_id]}
    colors = {"primary": "lime", "repeat": "magenta", "v2.1": "cyan"}
    polygons = {label: _accepted_annotations_for_component(record, component_id) for label, record in records.items()}
    figure, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(NAV_CONTEXT_COLORS[terrain_indices]); axes[0, 1].set_title("NAV terrain context")
    axes[0, 2].imshow(rgb); axes[0, 2].imshow(np.ma.masked_where(~candidate_mask, candidate_mask), cmap="YlOrBr", alpha=0.55); axes[0, 2].set_title("Semantic candidate reference only")
    for axis, label in zip(axes[1], ("primary", "repeat", "v2.1")):
        axis.imshow(rgb)
        axis.imshow(np.ma.masked_where(~candidate_mask, candidate_mask), cmap="YlOrBr", alpha=0.15)
        for annotation in polygons[label]:
            polygon = annotation["polygon"] + [annotation["polygon"][0]]
            axis.plot([point[0] for point in polygon], [point[1] for point in polygon], color=colors[label], linewidth=2.5)
        axis.set_title(f"{label} accepted extent")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(f"{target['target_id']} | historical overlays for forensic comparison only", fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def refresh_boundary_forensics(
    *, primary_state_path: Path, repeat_state_path: Path, v21_state_path: Path, boundary_state_path: Path, dataset_root: Path,
) -> None:
    """Regenerate forensic overlays without changing the human redraw state or source artifacts."""
    source_paths = {"primary": Path(primary_state_path), "repeat": Path(repeat_state_path), "v21": Path(v21_state_path)}
    boundary_state_path = Path(boundary_state_path)
    boundary = load_boundary_review_state(boundary_state_path)
    expected_hashes = {
        "primary": boundary["provenance"]["primary_state_sha256"],
        "repeat": boundary["provenance"]["repeat_state_sha256"],
        "v21": boundary["provenance"]["v21_state_sha256"],
    }
    if {label: sha256_file(path) for label, path in source_paths.items()} != expected_hashes:
        raise ValueError("Boundary forensic refresh requires the provenance-bound immutable source states.")
    primary, repeat, v21 = (load_review_state(source_paths[label]) for label in ("primary", "repeat", "v21"))
    for state in (primary, repeat, v21):
        validate_review_state(state)
    for target in boundary["targets"]:
        _render_forensic_overlay(
            primary=primary, repeat=repeat, v21=v21, target=target, dataset_root=Path(dataset_root),
            output_path=boundary_state_path.parent / "forensics" / _forensic_filename(target["target_id"]),
        )


def prepare_boundary_review(
    *, primary_state_path: Path, repeat_state_path: Path, v21_state_path: Path, component_candidates_csv: Path,
    target_manifest: Path, proposed_protocol_path: Path, dataset_root: Path, output_dir: Path,
) -> Path:
    """Create a blank, fixed-identity three-object boundary-redraw package."""
    paths = [Path(path) for path in (primary_state_path, repeat_state_path, v21_state_path)]
    primary, repeat, v21 = (load_review_state(path) for path in paths)
    for state in (primary, repeat, v21):
        validate_review_state(state)
    output_dir = Path(output_dir)
    component_candidates_csv = Path(component_candidates_csv)
    target_manifest = Path(target_manifest)
    proposed_protocol_path = Path(proposed_protocol_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing boundary-review artifacts: {output_dir}")
    if not component_candidates_csv.is_file() or not proposed_protocol_path.is_file():
        raise FileNotFoundError("Boundary review requires a component manifest and proposed v2.2 protocol.")
    rows = _selection_rows(target_manifest)
    metadata = v21.get("clarification_review", {})
    if v21.get("review_scope", {}).get("name") != "calibration_clarification" or any(v21["images"][image_id]["review_status"] != "reviewed" for image_id in v21["review_scope"]["image_ids"]):
        raise ValueError("Boundary review requires the completed isolated v2.1 clarification state.")
    if metadata.get("source_primary_state_sha256") != sha256_file(paths[0]) or metadata.get("source_repeat_state_sha256") != sha256_file(paths[1]):
        raise ValueError("v2.1 clarification provenance does not match immutable historic artifacts.")
    if component_candidates_csv and sha256_file(component_candidates_csv) != v21["component_review"]["component_manifest_sha256"]:
        raise ValueError("Boundary component manifest does not match v2.1 candidate provenance.")
    targets = [_target_record(v21, row) for row in rows]
    output_dir.mkdir(parents=True)
    copied_manifest = output_dir / target_manifest.name
    copied_components = output_dir / component_candidates_csv.name
    copied_protocol = output_dir / proposed_protocol_path.name
    for source, destination in ((target_manifest, copied_manifest), (component_candidates_csv, copied_components), (proposed_protocol_path, copied_protocol)):
        shutil.copyfile(source, destination)
    state = {
        "schema_version": BOUNDARY_REVIEW_SCHEMA_VERSION,
        "review_version": BOUNDARY_REVIEW_VERSION,
        "expert_splits_excluded": True,
        "review_scope": {"name": BOUNDARY_SCOPE_NAME, "target_ids": [target["target_id"] for target in targets], "source_manifest": copied_manifest.name, "source_manifest_sha256": sha256_file(copied_manifest)},
        "provenance": {
            "primary_state_sha256": sha256_file(paths[0]), "repeat_state_sha256": sha256_file(paths[1]), "v21_state_sha256": sha256_file(paths[2]),
            "proposed_protocol_sha256": sha256_file(copied_protocol), "component_manifest_sha256": sha256_file(copied_components),
            "historic_annotations_hidden": True, "v21_polygons_hidden": True,
        },
        "component_candidates_csv": copied_components.name,
        "proposed_protocol": {"version": BOUNDARY_REVIEW_VERSION, "path": str(copied_protocol), "sha256": sha256_file(copied_protocol)},
        "targets": targets,
    }
    state_path = output_dir / "review_state.json"
    save_boundary_review_state(state_path, state)
    for target in targets:
        _render_forensic_overlay(primary=primary, repeat=repeat, v21=v21, target=target, dataset_root=dataset_root, output_path=output_dir / "forensics" / _forensic_filename(target["target_id"]))
    _atomic_write_json(output_dir / "provenance.json", {
        **state["provenance"], "target_manifest": copied_manifest.name, "target_manifest_sha256": sha256_file(copied_manifest),
        "proposed_protocol": copied_protocol.name, "component_manifest": copied_components.name,
        "historic_annotations_hidden": True, "v21_polygons_hidden": True, "human_redraws_initialized": 0,
    })
    return state_path


def prepare_final_clarification_review(
    *, primary_state_path: Path, repeat_state_path: Path, v21_state_path: Path, boundary_state_path: Path,
    component_candidates_csv: Path, target_manifest: Path, proposed_protocol_path: Path, output_dir: Path,
) -> Path:
    """Create a blank, one-object whole-visible-object clarification package for the completed component-8 redraw."""
    source_paths = {"primary": Path(primary_state_path), "repeat": Path(repeat_state_path), "v21": Path(v21_state_path), "boundary": Path(boundary_state_path)}
    primary, repeat, v21 = (load_review_state(source_paths[label]) for label in ("primary", "repeat", "v21"))
    for state in (primary, repeat, v21):
        validate_review_state(state)
    boundary = load_boundary_review_state(source_paths["boundary"])
    if boundary["schema_version"] != BOUNDARY_REVIEW_SCHEMA_VERSION:
        raise ValueError("Final clarification must source the completed three-object v2.2 boundary review.")
    expected_hashes = {"primary_state_sha256": sha256_file(source_paths["primary"]), "repeat_state_sha256": sha256_file(source_paths["repeat"]), "v21_state_sha256": sha256_file(source_paths["v21"])}
    if any(boundary["provenance"][key] != value for key, value in expected_hashes.items()):
        raise ValueError("Completed v2.2 boundary review provenance does not match immutable source artifacts.")
    source_target = _target_by_id(boundary, FINAL_CLARIFICATION_TARGET_ID)
    if source_target["review_status"] != "redrawn" or source_target["object_identity_fixed"] != "accepted" or source_target["identity_escalation"]:
        raise ValueError("Final clarification requires the completed accepted component-8 redraw without escalation.")
    component_candidates_csv, target_manifest, proposed_protocol_path = (Path(path) for path in (component_candidates_csv, target_manifest, proposed_protocol_path))
    if not component_candidates_csv.is_file() or not proposed_protocol_path.is_file():
        raise FileNotFoundError("Final clarification requires a component manifest and proposed v2.2.1 protocol.")
    if "v2.2.1-visible-object-continuity-clarified-proposed" not in proposed_protocol_path.read_text(encoding="utf-8"):
        raise ValueError("Final clarification requires the approved proposed v2.2.1 whole-object protocol.")
    if sha256_file(component_candidates_csv) != boundary["provenance"]["component_manifest_sha256"]:
        raise ValueError("Final clarification component manifest does not match completed v2.2 provenance.")
    row = _final_selection_row(target_manifest)
    if row["stable_source_image_id"] != source_target["image_id"] or int(row["source_candidate_component_id"]) != source_target["source_candidate_component_id"] or row["v21_instance_id"] != source_target["v21_instance_id"]:
        raise ValueError("Final clarification manifest does not match the completed component-8 source target.")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing final-clarification artifacts: {output_dir}")
    output_dir.mkdir(parents=True)
    copied_manifest, copied_components, copied_protocol = (output_dir / path.name for path in (target_manifest, component_candidates_csv, proposed_protocol_path))
    for source, destination in ((target_manifest, copied_manifest), (component_candidates_csv, copied_components), (proposed_protocol_path, copied_protocol)):
        shutil.copyfile(source, destination)
    target = {
        **{key: source_target[key] for key in ("target_id", "image_id", "sequence_id", "image_path", "mask_path", "image_width", "image_height", "source_candidate_component_id", "v21_instance_id", "object_identity_fixed")},
        "boundary_question": row["boundary_question"], "review_status": "unreviewed", "reviewer": None, "polygon": None, "bbox": None,
        "reviewer_notes": "", "identity_escalation": False, "identity_escalation_note": "",
    }
    state = {
        "schema_version": FINAL_CLARIFICATION_SCHEMA_VERSION, "review_version": FINAL_CLARIFICATION_VERSION, "expert_splits_excluded": True,
        "review_scope": {"name": FINAL_CLARIFICATION_SCOPE_NAME, "target_ids": [target["target_id"]], "source_manifest": copied_manifest.name, "source_manifest_sha256": sha256_file(copied_manifest)},
        "provenance": {**expected_hashes, "source_boundary_state_sha256": sha256_file(source_paths["boundary"]), "proposed_protocol_sha256": sha256_file(copied_protocol), "component_manifest_sha256": sha256_file(copied_components), "historic_annotations_hidden": True, "v21_polygons_hidden": True, "v22_polygons_hidden": True},
        "component_candidates_csv": copied_components.name,
        "proposed_protocol": {"version": FINAL_CLARIFICATION_VERSION, "path": str(copied_protocol), "sha256": sha256_file(copied_protocol)},
        "source_v22_target": {"target_id": source_target["target_id"], "review_status": source_target["review_status"], "object_identity_fixed": source_target["object_identity_fixed"]},
        "targets": [target],
    }
    state_path = output_dir / "review_state.json"
    save_boundary_review_state(state_path, state)
    _atomic_write_json(output_dir / "provenance.json", {**state["provenance"], "target_manifest": copied_manifest.name, "target_manifest_sha256": sha256_file(copied_manifest), "proposed_protocol": copied_protocol.name, "component_manifest": copied_components.name, "historic_annotations_hidden": True, "v21_polygons_hidden": True, "v22_polygons_hidden": True, "human_redraws_initialized": 0})
    return state_path


class BoundaryReviewUI:
    """Full-image, polygon-only redraw UI with optional nonbinding proposal boxes."""

    def __init__(self, *, state_path: Path, dataset_root: Path, target_id: str | None, reviewer: str) -> None:
        self.state_path, self.dataset_root, self.reviewer = Path(state_path), Path(dataset_root), reviewer
        self.state = load_boundary_review_state(self.state_path)
        self.component_csv = self.state_path.parent / self.state["component_candidates_csv"]
        pending = [target["target_id"] for target in self.state["targets"] if target["review_status"] in {"unreviewed", "in_progress"}]
        self.target_id = target_id or (pending[0] if pending else None)
        if self.target_id is None:
            raise ValueError("All boundary targets are complete.")
        self.figure = plt.figure(figsize=(17, 9))
        self._load_target(self.target_id)

    def _load_target(self, target_id: str) -> None:
        self.target_id = target_id
        self.target = _target_by_id(self.state, target_id)
        self.polygon = copy.deepcopy(self.target["polygon"]) or []
        self.notes = self.target["reviewer_notes"]
        self.show_candidates = False
        self._render()

    def _render(self) -> None:
        if hasattr(self, "_click_connection"):
            self.figure.canvas.mpl_disconnect(self._click_connection)
        self.figure.clear()
        self.rgb_axis = self.figure.add_axes((0.02, 0.25, 0.46, 0.70))
        self.context_axis = self.figure.add_axes((0.52, 0.25, 0.46, 0.70))
        with Image.open(self.dataset_root / self.target["image_path"]) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        with Image.open(self.dataset_root / self.target["mask_path"]) as handle:
            mask = normalize_ai4mars_mask(np.asarray(handle, dtype=np.int64), self.dataset_root / self.target["mask_path"])
        terrain = NAV_CONTEXT_COLORS[np.where((mask >= 0) & (mask <= 3), mask, 4)]
        self.rgb_axis.imshow(rgb); self.context_axis.imshow(terrain)
        self.rgb_axis.set_title("Draw on full-resolution RGB; historical masks are hidden")
        self.context_axis.set_title("NAV terrain context; not a boundary target")
        for axis in (self.rgb_axis, self.context_axis): axis.axis("off")
        self._draw_overlays()
        self.figure.suptitle(f"Boundary-only redraw | {self.target_id} | accepted identity fixed | {self.target['boundary_question']}", fontsize=12)
        self._build_controls()
        self._click_connection = self.figure.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.figure.canvas.draw_idle()

    def _draw_overlays(self) -> None:
        if self.show_candidates:
            for row in _candidate_rows(self.component_csv, self.target["image_id"]):
                component_id = int(row["component_id"])
                color = "yellow" if component_id == self.target["source_candidate_component_id"] else "gray"
                box = Rectangle((int(row["bbox_left"]), int(row["bbox_top"])), int(row["bbox_width"]), int(row["bbox_height"]), fill=False, linewidth=1.5, edgecolor=color)
                self.rgb_axis.add_patch(box)
                self.rgb_axis.text(box.get_x(), box.get_y(), f"proposal {component_id}", color=color, fontsize=8)
        if self.polygon:
            closed = self.polygon + ([self.polygon[0]] if len(self.polygon) > 2 else [])
            self.rgb_axis.plot([point[0] for point in closed], [point[1] for point in closed], color="cyan", linewidth=2.5, marker="o")

    def _build_controls(self) -> None:
        candidates = CheckButtons(self.figure.add_axes((0.02, 0.12, 0.20, 0.07)), ("Show nonbinding proposals",), (False,))
        candidates.on_clicked(lambda _label: self._toggle_candidates())
        draw = Button(self.figure.add_axes((0.25, 0.13, 0.10, 0.05)), "Draw")
        draw.on_clicked(lambda _event: self._set_message("Click polygon vertices on the RGB panel. Pan/zoom in the Matplotlib toolbar is available; drawing is never clipped to a proposal box."))
        undo = Button(self.figure.add_axes((0.36, 0.13, 0.10, 0.05)), "Undo")
        undo.on_clicked(lambda _event: self._undo())
        notes = Button(self.figure.add_axes((0.47, 0.13, 0.10, 0.05)), "Notes")
        notes.on_clicked(lambda _event: self._edit_notes())
        save = Button(self.figure.add_axes((0.60, 0.13, 0.13, 0.05)), "Save redraw")
        save.on_clicked(lambda _event: self._save_redraw())
        escalate = Button(self.figure.add_axes((0.75, 0.13, 0.13, 0.05)), "Escalate identity")
        escalate.on_clicked(lambda _event: self._escalate())
        finish = Button(self.figure.add_axes((0.60, 0.05, 0.13, 0.05)), "Finish target")
        finish.on_clicked(lambda _event: self._finish())
        self.message_axis = self.figure.add_axes((0.02, 0.02, 0.54, 0.07)); self.message_axis.axis("off")
        self.message = self.message_axis.text(0, 0.9, "Draw only visible pixels of the fixed accepted object. Do not decide existence, split, or merge.", va="top", wrap=True)
        self.controls = [candidates, draw, undo, notes, save, escalate, finish]

    def _set_message(self, value: str, *, error: bool = False) -> None:
        self.message.set_text(value); self.message.set_color("crimson" if error else "black"); self.figure.canvas.draw_idle()

    def _toggle_candidates(self) -> None:
        self.show_candidates = not self.show_candidates
        self._render()

    def _on_canvas_click(self, event: Any) -> None:
        if event.inaxes is self.rgb_axis and event.xdata is not None and event.ydata is not None:
            self.polygon.append([float(event.xdata), float(event.ydata)])
            self._render()
            self._set_message(f"Polygon has {len(self.polygon)} point(s).")

    def _undo(self) -> None:
        if self.polygon: self.polygon.pop()
        self._render(); self._set_message(f"Polygon has {len(self.polygon)} point(s).")

    def _edit_notes(self) -> None:
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk(); root.withdraw()
            value = simpledialog.askstring("Boundary redraw notes", "Optional visible-edge note", initialvalue=self.notes, parent=root)
            root.destroy()
            if value is not None: self.notes = value
        except Exception as error:
            self._set_message(f"Unable to open notes dialog: {error}", error=True)

    def _save_redraw(self) -> None:
        try:
            if len(self.polygon) < 3: raise ValueError("A boundary redraw requires at least three polygon vertices.")
            updated = copy.deepcopy(self.state)
            record_boundary_redraw(updated, self.target_id, polygon=self.polygon, reviewer=self.reviewer, notes=self.notes)
            save_boundary_review_state(self.state_path, updated); self.state = updated; self.target = _target_by_id(self.state, self.target_id)
            self._set_message("Redraw saved. Finish target when the visible extent is final.")
        except ValueError as error: self._set_message(str(error), error=True)

    def _escalate(self) -> None:
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk(); root.withdraw()
            note = simpledialog.askstring("Identity escalation", "Why does the fixed target reference not identify a defensible object?", parent=root)
            root.destroy()
            updated = copy.deepcopy(self.state)
            record_identity_escalation(updated, self.target_id, reviewer=self.reviewer, note=note or "")
            save_boundary_review_state(self.state_path, updated); self.state = updated; self.target = _target_by_id(self.state, self.target_id)
            self._set_message("Identity escalation recorded; no annotation label was changed.")
        except ValueError as error: self._set_message(str(error), error=True)

    def _finish(self) -> None:
        try:
            updated = copy.deepcopy(self.state); finish_boundary_target(updated, self.target_id, reviewer=self.reviewer); save_boundary_review_state(self.state_path, updated); self.state = updated
            pending = [target["target_id"] for target in self.state["targets"] if target["review_status"] in {"unreviewed", "in_progress"}]
            if pending: self._load_target(pending[0])
            else: self._set_message("All boundary targets are complete. Stop before any broader review or protocol action.")
        except ValueError as error: self._set_message(str(error), error=True)

    def show(self) -> None:
        plt.show()


def activate_interactive_backend() -> None:
    """Select the GUI canvas before the reviewer constructs any figures."""
    try:
        plt.switch_backend("TkAgg")
    except ImportError as error:
        raise RuntimeError(
            "The boundary reviewer requires the Tk GUI backend. Run --interactive in a local desktop terminal with tkinter available."
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-final-clarification", action="store_true")
    parser.add_argument("--refresh-forensics", action="store_true")
    parser.add_argument("--primary-state-path", type=Path)
    parser.add_argument("--repeat-state-path", type=Path)
    parser.add_argument("--v21-state-path", type=Path)
    parser.add_argument("--component-candidates-csv", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--proposed-protocol-path", type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--target-id")
    parser.add_argument("--reviewer", default="single_researcher")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        required = (args.primary_state_path, args.repeat_state_path, args.v21_state_path, args.component_candidates_csv, args.target_manifest, args.proposed_protocol_path, args.output_dir)
        if any(value is None for value in required): raise ValueError("--prepare requires all source paths and --output-dir.")
        print(prepare_boundary_review(primary_state_path=args.primary_state_path, repeat_state_path=args.repeat_state_path, v21_state_path=args.v21_state_path, component_candidates_csv=args.component_candidates_csv, target_manifest=args.target_manifest, proposed_protocol_path=args.proposed_protocol_path, dataset_root=args.dataset_root, output_dir=args.output_dir))
    elif args.prepare_final_clarification:
        required = (args.primary_state_path, args.repeat_state_path, args.v21_state_path, args.state_path, args.component_candidates_csv, args.target_manifest, args.proposed_protocol_path, args.output_dir)
        if any(value is None for value in required): raise ValueError("--prepare-final-clarification requires source state paths, --state-path, component manifest, target manifest, protocol, and --output-dir.")
        print(prepare_final_clarification_review(primary_state_path=args.primary_state_path, repeat_state_path=args.repeat_state_path, v21_state_path=args.v21_state_path, boundary_state_path=args.state_path, component_candidates_csv=args.component_candidates_csv, target_manifest=args.target_manifest, proposed_protocol_path=args.proposed_protocol_path, output_dir=args.output_dir))
    elif args.refresh_forensics:
        required = (args.primary_state_path, args.repeat_state_path, args.v21_state_path, args.state_path)
        if any(value is None for value in required): raise ValueError("--refresh-forensics requires source state paths and --state-path.")
        refresh_boundary_forensics(primary_state_path=args.primary_state_path, repeat_state_path=args.repeat_state_path, v21_state_path=args.v21_state_path, boundary_state_path=args.state_path, dataset_root=args.dataset_root)
    elif args.interactive:
        if args.state_path is None: raise ValueError("--interactive requires --state-path.")
        activate_interactive_backend()
        BoundaryReviewUI(state_path=args.state_path, dataset_root=args.dataset_root, target_id=args.target_id, reviewer=args.reviewer).show()
    else:
        raise ValueError("Use --prepare or --interactive.")


if __name__ == "__main__":
    main()