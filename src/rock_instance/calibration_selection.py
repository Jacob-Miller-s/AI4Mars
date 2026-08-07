"""Deterministically select a train/val-only annotation-protocol calibration subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

from src.rock_instance.common import DEVELOPMENT_SPLITS, write_csv, write_json


CALIBRATION_FIELDS = [
    "calibration_rank", "pilot_rank", "stable_source_image_id", "split", "sequence_id", "image_path", "mask_path",
    "annotation_status", "selection_strata", "component_count", "big_rock_pixel_count", "bedrock_adjacent_pixels",
    "geometry_status", "selection_seed", "calibration_selection_seed", "calibration_strata", "selection_rationale",
]


def _rank(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"calibration:{seed}:{image_id}".encode("utf-8")).hexdigest()


def _categories(record: dict[str, str]) -> set[str]:
    strata = set(record["selection_strata"].split("|"))
    categories = set(strata)
    if int(record["component_count"]) >= 3:
        categories.add("fragmented_looking")
    return categories


def select_calibration_records(records: list[dict[str, str]], *, target_size: int = 24, seed: int = 42) -> list[dict[str, Any]]:
    """Select 20-30 pilot images with protocol stressors, never expert imagery."""
    if not 20 <= target_size <= 30:
        raise ValueError("Calibration target_size must be between 20 and 30.")
    if any(record.get("split") not in DEVELOPMENT_SPLITS for record in records):
        raise ValueError("Calibration selection accepts only train and val candidate records.")
    if any(record.get("annotation_status") != "candidate_unreviewed" for record in records):
        raise ValueError("Calibration candidates must remain unreviewed Sprint 0 pilot rows.")
    candidates = sorted(records, key=lambda record: (_rank(seed, record["stable_source_image_id"]), int(record["pilot_rank"])))
    if len({record["stable_source_image_id"] for record in candidates}) != len(candidates):
        raise ValueError("Pilot candidate manifest contains duplicate source image IDs.")
    if len(candidates) < target_size:
        raise ValueError("Pilot candidate manifest is smaller than the requested calibration subset.")
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    desired = (
        "isolated_candidate", "multiple_candidate_regions", "bedrock_big_rock_boundary",
        "very_large_component_candidate", "tiny_component_candidate", "border_truncation_candidate", "fragmented_looking",
    )
    for category in desired:
        quota = 3 if category == "border_truncation_candidate" else 4
        for record in candidates:
            if len(selected) == target_size or sum(category in _categories(item) for item in selected) >= quota:
                break
            if category in _categories(record) and record["stable_source_image_id"] not in selected_ids:
                selected.append(record)
                selected_ids.add(record["stable_source_image_id"])
    for record in candidates:
        if len(selected) == target_size:
            break
        if record["stable_source_image_id"] not in selected_ids:
            selected.append(record)
            selected_ids.add(record["stable_source_image_id"])
    return [
        {
            "calibration_rank": index,
            **record,
            "calibration_selection_seed": seed,
            "calibration_strata": "|".join(sorted(_categories(record) & set(desired))),
        }
        for index, record in enumerate(selected, start=1)
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-candidates-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = select_calibration_records(_read_csv(args.pilot_candidates_csv), target_size=args.target_size, seed=args.seed)
    write_csv(args.output_dir / "rock_instance_pilot_calibration_candidates.csv", records, CALIBRATION_FIELDS)
    write_json(
        args.output_dir / "rock_instance_pilot_calibration_summary.json",
        {
            "target_size": args.target_size,
            "selected_size": len(records),
            "selection_seed": args.seed,
            "expert_splits_excluded": True,
            "unique_sequences": len({record["sequence_id"] for record in records}),
            "strata_counts": {
                category: sum(category in record["calibration_strata"].split("|") for record in records)
                for category in sorted({category for record in records for category in record["calibration_strata"].split("|") if category})
            },
        },
    )


if __name__ == "__main__":
    main()