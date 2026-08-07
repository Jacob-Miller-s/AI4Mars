"""Inventory development-image stereo and geometry metadata without guessing pairs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.rock_instance.common import load_development_manifest_rows, require_development_splits, write_csv, write_json


INVENTORY_FIELDS = [
    "split",
    "stable_source_image_id",
    "sequence_id",
    "image_path",
    "mission",
    "rover",
    "camera",
    "camera_side",
    "image_file_status",
    "candidate_stereo_mate_path",
    "stereo_pairing_status",
    "range_depth_product_path",
    "range_depth_availability",
    "range_validity_mask_path",
    "range_validity_mask_availability",
    "calibration_geometry_metadata_availability",
    "usable_for_instance_pilot",
    "usable_for_stereo_geometry",
    "reason",
]


def _auxiliary_mask_path(dataset_root: Path, image_relative_path: str, product: str) -> Path | None:
    """Resolve documented MSL NAVCAM auxiliary masks using the EDR product marker."""
    image_path = Path(image_relative_path)
    if image_path.suffix.lower() not in {".jpg", ".jpeg"} or "EDR_" not in image_path.stem:
        return None
    auxiliary_stem = image_path.stem.replace("EDR_", f"{product}_", 1)
    return Path(dataset_root) / "msl" / "ncam" / "images" / product.lower().replace("rng", "rng-") / f"{auxiliary_stem}.png"


def _range_validity_mask_path(dataset_root: Path, image_relative_path: str) -> Path | None:
    """Resolve the release's documented binary RNG-30M validity mask, not depth."""
    image_path = Path(image_relative_path)
    if image_path.suffix.lower() not in {".jpg", ".jpeg"} or "EDR_" not in image_path.stem:
        return None
    auxiliary_stem = image_path.stem.replace("EDR_", "RNG_", 1)
    return Path(dataset_root) / "msl" / "ncam" / "images" / "rng-30m" / f"{auxiliary_stem}.png"


def _relative_if_present(path: Path | None, dataset_root: Path) -> str:
    if path is None or not path.is_file():
        return ""
    return path.relative_to(dataset_root).as_posix()


def inventory_records(dataset_root: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Create one conservative geometry-inventory record per validated development row."""
    records: list[dict[str, Any]] = []
    for row in rows:
        image_path = Path(dataset_root) / row["dataset_relative_image_path"]
        image_exists = image_path.is_file()
        range_mask = _range_validity_mask_path(dataset_root, row["dataset_relative_image_path"])
        range_mask_relative = _relative_if_present(range_mask, dataset_root)
        records.append(
            {
                "split": row["split"],
                "stable_source_image_id": row["stable_source_image_id"],
                "sequence_id": row["sequence_id"],
                "image_path": row["dataset_relative_image_path"],
                "mission": row["mission"],
                "rover": row["rover"],
                "camera": row["camera"],
                "camera_side": "unknown_no_explicit_manifest_metadata",
                "image_file_status": "present" if image_exists else "missing",
                "candidate_stereo_mate_path": "",
                "stereo_pairing_status": "unresolved_no_explicit_pair_metadata",
                "range_depth_product_path": "",
                "range_depth_availability": "unresolved_no_depth_product_metadata",
                "range_validity_mask_path": range_mask_relative,
                "range_validity_mask_availability": "present" if range_mask_relative else "missing",
                "calibration_geometry_metadata_availability": "unresolved_no_calibration_metadata",
                "usable_for_instance_pilot": image_exists,
                "usable_for_stereo_geometry": False,
                "reason": (
                    "No explicit stereo mate, depth product, or calibration metadata is present in the scoped manifest. "
                    "RNG-30M is recorded separately as a binary range-validity mask."
                ),
            }
        )
    return records


def summarize_records(records: list[dict[str, Any]], splits: tuple[str, ...]) -> dict[str, Any]:
    """Summarize inventory statuses without reinterpreting unresolved metadata."""
    counters = Counter(record["image_file_status"] for record in records)
    return {
        "scope": {"splits": list(splits), "expert_splits_excluded": True},
        "total_images": len(records),
        "images_by_split": dict(sorted(Counter(record["split"] for record in records).items())),
        "image_file_status": dict(sorted(counters.items())),
        "confirmed_stereo_mates": sum(record["stereo_pairing_status"] == "confirmed" for record in records),
        "confirmed_depth_products": sum(record["range_depth_availability"] == "confirmed" for record in records),
        "range_validity_masks_present": sum(record["range_validity_mask_availability"] == "present" for record in records),
        "sufficient_geometry_metadata": sum(record["usable_for_stereo_geometry"] for record in records),
        "unresolved_stereo_or_geometry": sum(record["stereo_pairing_status"].startswith("unresolved") for record in records),
        "invalid_or_missing_files": counters["missing"],
        "interpretation": (
            "The archive's RNG-30M files are binary masks for excluding ranges beyond 30 m, not metric depth products. "
            "No candidate stereo mate or geometry readiness is confirmed without explicit metadata."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = require_development_splits(args.splits)
    rows = load_development_manifest_rows(
        args.manifest_root,
        {"train": args.train_manifest, "val": args.val_manifest},
        splits,
    )
    records = inventory_records(args.dataset_root, rows)
    summary = summarize_records(records, splits)
    write_csv(args.output_dir / "stereo_inventory.csv", records, INVENTORY_FIELDS)
    write_json(args.output_dir / "stereo_inventory_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()