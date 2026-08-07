"""Summarize manual rock-instance pilot review progress without fabricating annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.rock_instance.annotations import load_review_state, polygon_to_mask, reviewed_annotations


def _calibration_image_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["stable_source_image_id"] for row in csv.DictReader(handle)}


def summarize_review_state(state: dict[str, Any], *, calibration_image_ids: set[str] | None = None) -> dict[str, Any]:
    """Summarize actual review decisions and leave burden estimates absent until review exists."""
    calibration_image_ids = calibration_image_ids or set()
    images = state["images"]
    reviewed = [image for image in images.values() if image["review_status"] == "reviewed"]
    annotations = list(reviewed_annotations(state))
    status_counts = Counter(annotation["annotation_status"] for annotation in annotations)
    accepted = [annotation for annotation in annotations if annotation["annotation_status"] == "accepted"]
    areas = []
    for annotation in accepted:
        image = images[annotation["image_id"]]
        areas.append(
            int(
                polygon_to_mask(
                    annotation["polygon"], image_width=image["image_width"], image_height=image["image_height"]
                ).sum()
            )
        )
    reviewed_calibration = [image for image in reviewed if image["image_id"] in calibration_image_ids]
    report: dict[str, Any] = {
        "schema_version": state["schema_version"],
        "pilot_id": state["pilot_id"],
        "images_total": len(images),
        "images_reviewed": len(reviewed),
        "images_deferred": sum(image["review_status"] == "deferred" for image in images.values()),
        "images_remaining": len(images) - len(reviewed),
        "accepted_rock_instances": status_counts["accepted"],
        "rejected_bedrock_candidates": status_counts["rejected_bedrock"],
        "rejected_noise_candidates": status_counts["rejected_noise"],
        "uncertain_objects": status_counts["uncertain"],
        "split_required_cases": status_counts["split_required"],
        "merge_required_cases": status_counts["merge_required"],
        "truncated_instances": sum(annotation["truncated"] for annotation in annotations),
        "occluded_instances": sum(annotation["occluded"] for annotation in annotations),
        "accepted_instances_per_reviewed_image": len(accepted) / len(reviewed) if reviewed else None,
        "accepted_instance_area_pixels": {
            "count": len(areas), "min": min(areas, default=None), "median": float(np.median(areas)) if areas else None,
            "max": max(areas, default=None),
        },
        "sequence_coverage": {
            "reviewed_sequences": len({image["sequence_id"] for image in reviewed}),
            "total_sequences": len({image["sequence_id"] for image in images.values()}),
        },
        "calibration": {
            "images_total": len(calibration_image_ids),
            "images_reviewed": len(reviewed_calibration),
            "annotation_burden_estimate": None,
            "note": "No burden estimate is produced until at least one calibration image has been reviewed.",
        },
    }
    if reviewed_calibration:
        calibration_accepted = sum(
            annotation["annotation_status"] == "accepted"
            for image in reviewed_calibration for annotation in image["annotations"]
        )
        report["calibration"]["annotation_burden_estimate"] = {
            "accepted_instances_per_reviewed_calibration_image": calibration_accepted / len(reviewed_calibration),
            "projected_accepted_instances_at_150_images": calibration_accepted / len(reviewed_calibration) * len(images),
            "interpretation": "Preliminary extrapolation from reviewed calibration images, not a ground-truth count or time estimate.",
        }
        report["calibration"]["note"] = "Estimate is preliminary because calibration images are intentionally enriched for difficult cases."
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_review_state(load_review_state(args.state_path), calibration_image_ids=_calibration_image_ids(args.calibration_manifest))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()