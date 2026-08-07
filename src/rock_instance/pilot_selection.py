"""Select a deterministic, development-only rock-instance review pilot manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.rock_instance.common import write_csv, write_json


PILOT_FIELDS = [
    "pilot_rank", "stable_source_image_id", "split", "sequence_id", "image_path", "mask_path",
    "annotation_status", "selection_strata", "component_count", "big_rock_pixel_count",
    "bedrock_adjacent_pixels", "geometry_status", "selection_seed", "selection_rationale",
]


def _stable_rank(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strata(record: dict[str, str], geometry: dict[str, str]) -> list[str]:
    labels: list[str] = []
    if int(record["component_count"]) == 1 and not any(
        record[name].lower() == "true" for name in ("has_border_component", "has_very_large_component", "has_unusual_aspect_ratio")
    ):
        labels.append("isolated_candidate")
    if int(record["component_count"]) > 1:
        labels.append("multiple_candidate_regions")
    if int(record["bedrock_adjacent_pixels"]) > 0:
        labels.append("bedrock_big_rock_boundary")
    if record["has_border_component"].lower() == "true":
        labels.append("border_truncation_candidate")
    if record["has_very_large_component"].lower() == "true":
        labels.append("very_large_component_candidate")
    if record["has_tiny_component"].lower() == "true":
        labels.append("tiny_component_candidate")
    if geometry["usable_for_stereo_geometry"].lower() == "true":
        labels.append("confirmed_geometry")
    else:
        labels.append("rgb_only_or_geometry_unresolved")
    return labels


def select_pilot_records(
    image_records: list[dict[str, str]],
    geometry_records: list[dict[str, str]],
    *,
    target_size: int,
    seed: int,
    max_per_sequence: int = 1,
) -> list[dict[str, Any]]:
    """Select candidate rows with a deterministic diversity cap, never annotations."""
    if not 100 <= target_size <= 200:
        raise ValueError("target_size must be between 100 and 200 for the proposed pilot.")
    if max_per_sequence < 1:
        raise ValueError("max_per_sequence must be at least 1.")
    geometry_by_id = {record["stable_source_image_id"]: record for record in geometry_records}
    eligible: list[dict[str, str]] = []
    for record in image_records:
        if int(record["component_count"]) == 0:
            continue
        geometry = geometry_by_id.get(record["stable_source_image_id"])
        if geometry is None:
            raise ValueError(f"Inventory record missing for {record['stable_source_image_id']}.")
        if geometry["split"] != record["split"]:
            raise ValueError(f"Inventory split mismatch for {record['stable_source_image_id']}.")
        eligible.append(record)
    if len(eligible) < target_size:
        raise ValueError(f"Only {len(eligible)} Big Rock candidate images are available for target_size={target_size}.")

    ordered = sorted(eligible, key=lambda record: (-int(record["manual_review_priority"]), _stable_rank(seed, record["stable_source_image_id"])))
    selected: list[dict[str, Any]] = []
    selected_sequences: defaultdict[str, int] = defaultdict(int)

    def add_record(record: dict[str, str]) -> bool:
        if len(selected) == target_size or selected_sequences[record["sequence_id"]] >= max_per_sequence:
            return False
        geometry = geometry_by_id[record["stable_source_image_id"]]
        strata = _strata(record, geometry)
        selected_sequences[record["sequence_id"]] += 1
        selected.append(
            {
                "pilot_rank": len(selected) + 1,
                "stable_source_image_id": record["stable_source_image_id"],
                "split": record["split"],
                "sequence_id": record["sequence_id"],
                "image_path": record["image_path"],
                "mask_path": record["mask_path"],
                "annotation_status": "candidate_unreviewed",
                "selection_strata": "|".join(strata),
                "component_count": int(record["component_count"]),
                "big_rock_pixel_count": int(record["big_rock_pixel_count"]),
                "bedrock_adjacent_pixels": int(record["bedrock_adjacent_pixels"]),
                "geometry_status": geometry["stereo_pairing_status"],
                "selection_seed": seed,
                "selection_rationale": "deterministic development-only candidate selection; requires manual instance review",
            }
        )
        return True

    target_quotas = {
        "isolated_candidate": round(target_size * 0.20),
        "multiple_candidate_regions": round(target_size * 0.20),
        "bedrock_big_rock_boundary": round(target_size * 0.20),
        "border_truncation_candidate": round(target_size * 0.05),
        "very_large_component_candidate": round(target_size * 0.20),
    }
    target_quotas["tiny_component_candidate"] = target_size - sum(target_quotas.values())
    for stratum, quota in target_quotas.items():
        accepted = 0
        for record in ordered:
            if stratum not in _strata(record, geometry_by_id[record["stable_source_image_id"]]):
                continue
            accepted += add_record(record)
            if accepted == quota:
                break
    for record in ordered:
        if len(selected) == target_size:
            break
        add_record(record)
    if len(selected) != target_size:
        raise ValueError("Sequence diversity cap prevented the requested pilot size; lower --max-per-sequence only with protocol review.")
    return selected


def summarize_pilot(records: list[dict[str, Any]], *, target_size: int, seed: int) -> dict[str, Any]:
    """Describe the proposed candidate pilot without implying manual labels exist."""
    return {
        "target_size": target_size,
        "selected_size": len(records),
        "selection_seed": seed,
        "expert_splits_excluded": True,
        "annotation_statement": "All rows are candidate_unreviewed; this manifest contains no instance ground truth.",
        "unique_sequences": len({record["sequence_id"] for record in records}),
        "split_counts": {split: sum(record["split"] == split for record in records) for split in ("train", "val")},
        "strata_counts": {
            label: sum(label in record["selection_strata"].split("|") for record in records)
            for label in sorted({label for record in records for label in record["selection_strata"].split("|")})
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-images-csv", required=True, type=Path)
    parser.add_argument("--stereo-inventory-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-sequence", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = select_pilot_records(
        _read_csv(args.component_images_csv),
        _read_csv(args.stereo_inventory_csv),
        target_size=args.target_size,
        seed=args.seed,
        max_per_sequence=args.max_per_sequence,
    )
    summary = summarize_pilot(records, target_size=args.target_size, seed=args.seed)
    write_csv(args.output_dir / "rock_instance_pilot_candidates.csv", records, PILOT_FIELDS)
    write_json(args.output_dir / "rock_instance_pilot_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()