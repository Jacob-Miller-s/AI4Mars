"""Compare completed v2.2 boundary redraws with immutable historical geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.rock_instance.annotations import annotation_component_ids, load_review_state, polygon_to_mask, sha256_file
from src.rock_instance.boundary_review import load_boundary_review_state
from src.rock_instance.intra_rater_consistency import mask_iou


def _accepted_for_component(state: dict[str, Any], image_id: str, component_id: int) -> list[dict[str, Any]]:
    return [
        annotation for annotation in state["images"][image_id]["annotations"]
        if annotation["annotation_status"] == "accepted" and component_id in annotation_component_ids(annotation)
    ]


def analyze_boundary_redraw(
    primary_state_path: Path, repeat_state_path: Path, v21_state_path: Path, boundary_state_path: Path,
) -> dict[str, Any]:
    """Compare completed redraw geometry without modifying any review artifact."""
    paths = [Path(path) for path in (primary_state_path, repeat_state_path, v21_state_path, boundary_state_path)]
    before_hashes = {str(path): sha256_file(path) for path in paths}
    primary, repeat, v21 = (load_review_state(path) for path in paths[:3])
    boundary = load_boundary_review_state(paths[3])
    provenance = boundary["provenance"]
    expected = {
        "primary_state_sha256": sha256_file(paths[0]),
        "repeat_state_sha256": sha256_file(paths[1]),
        "v21_state_sha256": sha256_file(paths[2]),
    }
    if any(provenance[key] != value for key, value in expected.items()):
        raise ValueError("Boundary redraw provenance does not match the supplied immutable artifacts.")
    incomplete = [target["target_id"] for target in boundary["targets"] if target["review_status"] != "redrawn"]
    if incomplete:
        raise ValueError(f"Boundary consistency analysis requires three completed redraws without identity escalation: {incomplete}")
    comparisons = []
    for target in boundary["targets"]:
        image_id, component_id = target["image_id"], target["source_candidate_component_id"]
        new_mask = polygon_to_mask(target["polygon"], image_width=target["image_width"], image_height=target["image_height"])
        for label, state in (("primary", primary), ("repeat", repeat), ("v2.1", v21)):
            annotations = _accepted_for_component(state, image_id, component_id)
            if label == "v2.1":
                annotations = [annotation for annotation in annotations if annotation["instance_id"] == target["v21_instance_id"]]
            for annotation in annotations:
                historic_mask = polygon_to_mask(annotation["polygon"], image_width=target["image_width"], image_height=target["image_height"])
                comparisons.append({
                    "target_id": target["target_id"], "image_id": image_id, "comparison": f"v2.2_to_{label}",
                    "v22_area_pixels": int(new_mask.sum().item()), "v22_bbox": target["bbox"],
                    "historical_instance_id": annotation["instance_id"], "historical_area_pixels": int(historic_mask.sum().item()),
                    "historical_bbox": annotation["bbox"], "mask_iou": mask_iou(new_mask, historic_mask),
                })
    report = {
        "analysis_type": "v2.2 boundary consistency",
        "provenance": {**expected, "boundary_state_sha256": sha256_file(paths[3]), "artifacts_distinct": len({path.resolve() for path in paths}) == 4},
        "targets": [{key: target[key] for key in ("target_id", "image_id", "source_candidate_component_id", "v21_instance_id", "polygon", "bbox", "reviewer_notes")} for target in boundary["targets"]],
        "comparisons": comparisons,
        "qualitative_review_required": "Assess whether each redraw follows the v2.2 visible-image-plane rule. No historic mask is ground truth and no universal IoU threshold decides the recommendation.",
        "CALIBRATION_PROTOCOL_RECOMMENDATION": "CLARIFY_AGAIN",
    }
    if before_hashes != {str(path): sha256_file(path) for path in paths}:
        raise RuntimeError("Boundary consistency analysis unexpectedly modified a review artifact.")
    return report


def write_analysis_outputs(report: dict[str, Any], output_dir: Path, markdown_path: Path) -> None:
    output_dir = Path(output_dir); markdown_path = Path(markdown_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "boundary_consistency.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# v2.2 Boundary Consistency", "", "No historic polygon is treated as ground truth.", "", "| Target | Comparison | v2.2 area | Historic area | Mask IoU |", "| --- | --- | ---: | ---: | ---: |"]
    lines.extend(f"| {row['target_id']} | {row['comparison']} | {row['v22_area_pixels']} | {row['historical_area_pixels']} | {row['mask_iou']:.4f} |" for row in report["comparisons"])
    lines.extend(["", "## Required Qualitative Review", "", f"- {report['qualitative_review_required']}", "", "`CALIBRATION_PROTOCOL_RECOMMENDATION = CLARIFY_AGAIN`", ""])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-state-path", required=True, type=Path)
    parser.add_argument("--repeat-state-path", required=True, type=Path)
    parser.add_argument("--v21-state-path", required=True, type=Path)
    parser.add_argument("--boundary-state-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--markdown-path", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_boundary_redraw(args.primary_state_path, args.repeat_state_path, args.v21_state_path, args.boundary_state_path)
    write_analysis_outputs(report, args.output_dir, args.markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()