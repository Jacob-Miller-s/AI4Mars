"""Analyze a completed one-object v2.2.1 clarification without mutating review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.rock_instance.annotations import annotation_component_ids, load_review_state, polygon_to_mask, sha256_file
from src.rock_instance.boundary_review import (
    FINAL_CLARIFICATION_SCHEMA_VERSION,
    FINAL_CLARIFICATION_TARGET_ID,
    FINAL_CLARIFICATION_VERSION,
    BOUNDARY_REVIEW_SCHEMA_VERSION,
    load_boundary_review_state,
)
from src.rock_instance.intra_rater_consistency import mask_iou


def _accepted_for_component(state: dict[str, Any], image_id: str, component_id: int) -> list[dict[str, Any]]:
    return [
        annotation for annotation in state["images"][image_id]["annotations"]
        if annotation["annotation_status"] == "accepted" and component_id in annotation_component_ids(annotation)
    ]


def _completed_target(state: dict[str, Any], *, schema_version: str, target_id: str, isolated_scope: bool) -> dict[str, Any]:
    scope_target_ids = state.get("review_scope", {}).get("target_ids", [])
    if state.get("schema_version") != schema_version or target_id not in scope_target_ids:
        raise ValueError("Final consistency analysis received an invalid isolated review scope.")
    if isolated_scope and scope_target_ids != [target_id]:
        raise ValueError("Final consistency analysis requires exactly the scoped component-8 target.")
    targets = [target for target in state["targets"] if target["target_id"] == target_id]
    if len(targets) != 1:
        raise ValueError("Final consistency analysis requires exactly one component-8 target record.")
    target = targets[0]
    if target["review_status"] != "redrawn" or target["object_identity_fixed"] != "accepted" or target["identity_escalation"]:
        raise ValueError("Final consistency analysis requires a completed fixed-accepted redraw without escalation.")
    return target


def _comparison_row(label: str, target: dict[str, Any], new_mask: Any, previous_polygon: list[list[float]], previous_bbox: list[int]) -> dict[str, Any]:
    previous_mask = polygon_to_mask(previous_polygon, image_width=target["image_width"], image_height=target["image_height"])
    new_area = int(new_mask.sum().item())
    previous_area = int(previous_mask.sum().item())
    return {
        "comparison": f"v2.2.1_to_{label}",
        "v221_area_pixels": new_area,
        "v221_bbox": target["bbox"],
        "previous_area_pixels": previous_area,
        "previous_bbox": previous_bbox,
        "mask_iou": mask_iou(new_mask, previous_mask),
        "v221_to_previous_area_ratio": new_area / previous_area,
    }


def analyze_final_clarification(
    primary_state_path: Path, repeat_state_path: Path, v21_state_path: Path, v22_state_path: Path, final_state_path: Path,
) -> dict[str, Any]:
    """Compare a completed final redraw against prior evidence while preserving every source artifact."""
    paths = {
        "primary": Path(primary_state_path), "repeat": Path(repeat_state_path), "v2.1": Path(v21_state_path),
        "v2.2": Path(v22_state_path), "v2.2.1": Path(final_state_path),
    }
    before_hashes = {label: sha256_file(path) for label, path in paths.items()}
    primary, repeat, v21 = (load_review_state(paths[label]) for label in ("primary", "repeat", "v2.1"))
    v22 = load_boundary_review_state(paths["v2.2"])
    final = load_boundary_review_state(paths["v2.2.1"])
    if final.get("schema_version") != FINAL_CLARIFICATION_SCHEMA_VERSION or final.get("review_version") != FINAL_CLARIFICATION_VERSION:
        raise ValueError("Final consistency analysis requires the proposed v2.2.1 final clarification artifact.")
    target = _completed_target(final, schema_version=FINAL_CLARIFICATION_SCHEMA_VERSION, target_id=FINAL_CLARIFICATION_TARGET_ID, isolated_scope=True)
    previous_target = _completed_target(v22, schema_version=BOUNDARY_REVIEW_SCHEMA_VERSION, target_id=FINAL_CLARIFICATION_TARGET_ID, isolated_scope=False)
    expected_hashes = {
        "primary_state_sha256": before_hashes["primary"], "repeat_state_sha256": before_hashes["repeat"],
        "v21_state_sha256": before_hashes["v2.1"], "source_boundary_state_sha256": before_hashes["v2.2"],
    }
    if any(final["provenance"][key] != value for key, value in expected_hashes.items()):
        raise ValueError("Final clarification provenance does not match the supplied immutable artifacts.")
    image_id, component_id = target["image_id"], target["source_candidate_component_id"]
    if previous_target["image_id"] != image_id or previous_target["source_candidate_component_id"] != component_id:
        raise ValueError("Final clarification does not identify the same v2.2 component-8 source target.")
    new_mask = polygon_to_mask(target["polygon"], image_width=target["image_width"], image_height=target["image_height"])
    comparisons = []
    for label, state in (("primary", primary), ("repeat", repeat), ("v2.1", v21)):
        annotations = _accepted_for_component(state, image_id, component_id)
        if label == "v2.1":
            annotations = [annotation for annotation in annotations if annotation["instance_id"] == target["v21_instance_id"]]
        if len(annotations) != 1:
            raise ValueError(f"Final consistency analysis requires one accepted {label} annotation for component 8.")
        annotation = annotations[0]
        comparisons.append(_comparison_row(label, target, new_mask, annotation["polygon"], annotation["bbox"]))
    comparisons.append(_comparison_row("v2.2", target, new_mask, previous_target["polygon"], previous_target["bbox"]))
    report = {
        "analysis_type": "v2.2.1 final whole-object clarification consistency",
        "provenance": {**before_hashes, "artifacts_distinct": len({path.resolve() for path in paths.values()}) == 5},
        "target": {key: target[key] for key in ("target_id", "image_id", "source_candidate_component_id", "v21_instance_id", "polygon", "bbox", "reviewer_notes")},
        "comparisons": comparisons,
        "qualitative_review_required": "Evaluate the redraw against the approved whole-object-versus-face rule. No prior annotation is ground truth and no universal IoU threshold decides compliance.",
        "CALIBRATION_PROTOCOL_RECOMMENDATION": "CLARIFY_AGAIN",
    }
    if before_hashes != {label: sha256_file(path) for label, path in paths.items()}:
        raise RuntimeError("Final consistency analysis unexpectedly modified a review artifact.")
    return report


def render_comparison_overlay(report: dict[str, Any], primary_state_path: Path, repeat_state_path: Path, v21_state_path: Path, v22_state_path: Path, output_path: Path, dataset_root: Path) -> None:
    target = report["target"]
    image_id, component_id = target["image_id"], target["source_candidate_component_id"]
    primary, repeat, v21 = (load_review_state(Path(path)) for path in (primary_state_path, repeat_state_path, v21_state_path))
    v22 = load_boundary_review_state(Path(v22_state_path))
    prior_polygons = {}
    for label, state in (("primary", primary), ("repeat", repeat), ("v2.1", v21)):
        annotations = _accepted_for_component(state, image_id, component_id)
        if label == "v2.1":
            annotations = [annotation for annotation in annotations if annotation["instance_id"] == target["v21_instance_id"]]
        prior_polygons[label] = annotations[0]["polygon"]
    prior_polygons["v2.2"] = _completed_target(v22, schema_version=BOUNDARY_REVIEW_SCHEMA_VERSION, target_id=target["target_id"], isolated_scope=False)["polygon"]
    prior_polygons["v2.2.1"] = target["polygon"]
    with Image.open(Path(dataset_root) / primary["images"][image_id]["image_path"]) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = {"primary": "#55d66b", "repeat": "#e95acb", "v2.1": "#24d5e8", "v2.2": "#ffd23f", "v2.2.1": "#ff5c4d"}
    all_points = [point for polygon in prior_polygons.values() for point in polygon]
    left, right = max(0, int(min(point[0] for point in all_points)) - 35), min(rgb.shape[1], int(max(point[0] for point in all_points)) + 36)
    top, bottom = max(0, int(min(point[1] for point in all_points)) - 35), min(rgb.shape[0], int(max(point[1] for point in all_points)) + 36)
    for axis, zoomed in ((axes[0], False), (axes[1], True)):
        axis.imshow(rgb)
        for label, polygon in prior_polygons.items():
            closed = polygon + [polygon[0]]
            axis.plot([point[0] for point in closed], [point[1] for point in closed], color=colors[label], linewidth=2.2)
        if zoomed:
            axis.set_xlim(left, right)
            axis.set_ylim(bottom, top)
            axis.set_title("Whole-object comparison (zoom)")
        else:
            axis.set_title("Full RGB")
        axis.axis("off")
    handles = [plt.Line2D([0], [0], color=colors[label], linewidth=3, label=label) for label in colors]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    figure.suptitle(f"{target['target_id']} | outlines only; no prior pass is ground truth", fontsize=13)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_analysis_outputs(report: dict[str, Any], output_dir: Path, markdown_path: Path) -> None:
    output_dir = Path(output_dir)
    markdown_path = Path(markdown_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_clarification_consistency.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# v2.2.1 Final Whole-Object Clarification Consistency", "", "No prior annotation is treated as ground truth.", "", "| Comparison | v2.2.1 area | Prior area | Area ratio | Mask IoU |", "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {row['comparison']} | {row['v221_area_pixels']} | {row['previous_area_pixels']} | {row['v221_to_previous_area_ratio']:.4f} | {row['mask_iou']:.4f} |" for row in report["comparisons"])
    lines.extend(["", "## Required Qualitative Review", "", f"- {report['qualitative_review_required']}", "", "`CALIBRATION_PROTOCOL_RECOMMENDATION = CLARIFY_AGAIN`", ""])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-state-path", required=True, type=Path)
    parser.add_argument("--repeat-state-path", required=True, type=Path)
    parser.add_argument("--v21-state-path", required=True, type=Path)
    parser.add_argument("--v22-state-path", required=True, type=Path)
    parser.add_argument("--final-state-path", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--markdown-path", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_final_clarification(args.primary_state_path, args.repeat_state_path, args.v21_state_path, args.v22_state_path, args.final_state_path)
    write_analysis_outputs(report, args.output_dir, args.markdown_path)
    render_comparison_overlay(report, args.primary_state_path, args.repeat_state_path, args.v21_state_path, args.v22_state_path, args.output_dir / "component-8_final_comparison.png", args.dataset_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()