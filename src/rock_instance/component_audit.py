"""Audit semantic Big Rock components as review candidates, never instance ground truth."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.dataset import normalize_ai4mars_mask
from src.rock_instance.common import load_development_manifest_rows, require_development_splits, write_csv, write_json


BIG_ROCK_CLASS_ID = 3
BEDROCK_CLASS_ID = 1
CONNECTIVITY = 8
COMPONENT_FIELDS = [
    "split", "stable_source_image_id", "sequence_id", "image_path", "mask_path", "component_id",
    "area_pixels", "bbox_left", "bbox_top", "bbox_width", "bbox_height", "centroid_x", "centroid_y",
    "aspect_ratio", "border_touching", "tiny_component", "very_large_component", "unusual_aspect_ratio",
]
IMAGE_FIELDS = [
    "split", "stable_source_image_id", "sequence_id", "image_path", "mask_path", "component_count",
    "big_rock_pixel_count", "bedrock_adjacent_pixels", "multiple_components", "nearby_components",
    "fragmentation_proxy", "has_tiny_component", "has_very_large_component", "has_border_component",
    "has_unusual_aspect_ratio", "manual_review_priority", "manual_review_reasons",
]


def _is_border_touching(left: int, top: int, width: int, height: int, image_width: int, image_height: int) -> bool:
    return left == 0 or top == 0 or left + width == image_width or top + height == image_height


def _boxes_are_nearby(first: tuple[int, int, int, int], second: tuple[int, int, int, int], gap_pixels: int = 4) -> bool:
    first_left, first_top, first_width, first_height = first
    second_left, second_top, second_width, second_height = second
    horizontal_gap = max(0, max(first_left, second_left) - min(first_left + first_width, second_left + second_width))
    vertical_gap = max(0, max(first_top, second_top) - min(first_top + first_height, second_top + second_height))
    return horizontal_gap <= gap_pixels and vertical_gap <= gap_pixels


def _bedrock_adjacency(mask: np.ndarray) -> int:
    big_rock = (mask == BIG_ROCK_CLASS_ID).astype(np.uint8)
    dilated = cv2.dilate(big_rock, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return int(np.count_nonzero((dilated > 0) & (mask == BEDROCK_CLASS_ID)))


def component_records_for_mask(
    mask: np.ndarray,
    row: dict[str, str],
    *,
    connectivity: int = CONNECTIVITY,
    tiny_area_pixels: int = 64,
    large_area_pixels: int = 50_000,
    unusual_aspect_ratio: float = 4.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure semantic components and image-level review signals for one source mask."""
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8.")
    big_rock = (mask == BIG_ROCK_CLASS_ID).astype(np.uint8)
    component_total, _, stats, centroids = cv2.connectedComponentsWithStats(big_rock, connectivity=connectivity)
    image_height, image_width = mask.shape
    components: list[dict[str, Any]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for component_id in range(1, component_total):
        left, top, width, height, area = (int(value) for value in stats[component_id])
        aspect_ratio = float(width / height) if height else float("inf")
        record = {
            "split": row["split"],
            "stable_source_image_id": row["stable_source_image_id"],
            "sequence_id": row["sequence_id"],
            "image_path": row["dataset_relative_image_path"],
            "mask_path": row["dataset_relative_mask_path"],
            "component_id": component_id,
            "area_pixels": area,
            "bbox_left": left,
            "bbox_top": top,
            "bbox_width": width,
            "bbox_height": height,
            "centroid_x": round(float(centroids[component_id][0]), 6),
            "centroid_y": round(float(centroids[component_id][1]), 6),
            "aspect_ratio": round(aspect_ratio, 6),
            "border_touching": _is_border_touching(left, top, width, height, image_width, image_height),
            "tiny_component": area <= tiny_area_pixels,
            "very_large_component": area >= large_area_pixels,
            "unusual_aspect_ratio": aspect_ratio >= unusual_aspect_ratio or aspect_ratio <= 1 / unusual_aspect_ratio,
        }
        components.append(record)
        boxes.append((left, top, width, height))

    nearby_components = any(
        _boxes_are_nearby(first, second)
        for index, first in enumerate(boxes)
        for second in boxes[index + 1 :]
    )
    reasons: list[str] = []
    if len(components) > 1:
        reasons.append("multiple_semantic_components")
    if nearby_components:
        reasons.append("nearby_separate_components")
    if len(components) >= 3:
        reasons.append("fragmentation_proxy")
    if any(component["very_large_component"] for component in components):
        reasons.append("very_large_component")
    if any(component["tiny_component"] for component in components):
        reasons.append("tiny_component")
    if any(component["border_touching"] for component in components):
        reasons.append("border_truncation_candidate")
    if any(component["unusual_aspect_ratio"] for component in components):
        reasons.append("unusual_aspect_ratio")
    bedrock_adjacent_pixels = _bedrock_adjacency(mask)
    if bedrock_adjacent_pixels:
        reasons.append("bedrock_big_rock_boundary")
    priority = (
        5 * nearby_components
        + 4 * (len(components) > 1)
        + 3 * (len(components) >= 3)
        + 3 * any(component["very_large_component"] for component in components)
        + 2 * any(component["border_touching"] for component in components)
        + 2 * any(component["unusual_aspect_ratio"] for component in components)
        + int(bool(bedrock_adjacent_pixels))
        + int(any(component["tiny_component"] for component in components))
    )
    image_record = {
        "split": row["split"],
        "stable_source_image_id": row["stable_source_image_id"],
        "sequence_id": row["sequence_id"],
        "image_path": row["dataset_relative_image_path"],
        "mask_path": row["dataset_relative_mask_path"],
        "component_count": len(components),
        "big_rock_pixel_count": int(big_rock.sum()),
        "bedrock_adjacent_pixels": bedrock_adjacent_pixels,
        "multiple_components": len(components) > 1,
        "nearby_components": nearby_components,
        "fragmentation_proxy": len(components) >= 3,
        "has_tiny_component": any(component["tiny_component"] for component in components),
        "has_very_large_component": any(component["very_large_component"] for component in components),
        "has_border_component": any(component["border_touching"] for component in components),
        "has_unusual_aspect_ratio": any(component["unusual_aspect_ratio"] for component in components),
        "manual_review_priority": priority,
        "manual_review_reasons": "|".join(reasons),
    }
    return components, image_record


def audit_records(
    dataset_root: Path,
    rows: list[dict[str, str]],
    **thresholds: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read development masks and return deterministic component and image records."""
    components: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for row in rows:
        mask_path = Path(dataset_root) / row["dataset_relative_mask_path"]
        if not mask_path.is_file():
            raise FileNotFoundError(f"Manifest mask is missing: {mask_path}")
        with Image.open(mask_path) as mask_file:
            mask = normalize_ai4mars_mask(np.asarray(mask_file, dtype=np.int64), mask_path)
        image_components, image_record = component_records_for_mask(mask, row, **thresholds)
        components.extend(image_components)
        images.append(image_record)
    return components, images


def review_candidates(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank only candidate images; the ranking is a review queue, not annotation truth."""
    return sorted(
        (record for record in image_records if record["manual_review_reasons"]),
        key=lambda record: (-record["manual_review_priority"], record["split"], record["sequence_id"], record["stable_source_image_id"]),
    )


def summarize_audit(component_records: list[dict[str, Any]], image_records: list[dict[str, Any]], splits: tuple[str, ...]) -> dict[str, Any]:
    """Provide deterministic aggregate counts while preserving candidate-only semantics."""
    areas = [record["area_pixels"] for record in component_records]
    return {
        "scope": {"splits": list(splits), "expert_splits_excluded": True},
        "connectivity": CONNECTIVITY,
        "candidate_semantic_class": {"id": BIG_ROCK_CLASS_ID, "name": "big_rock"},
        "ground_truth_statement": "Connected components are review candidates only and are not validated rock instances.",
        "images_total": len(image_records),
        "images_with_big_rock_candidates": sum(record["component_count"] > 0 for record in image_records),
        "components_total": len(component_records),
        "components_by_split": dict(sorted(Counter(record["split"] for record in component_records).items())),
        "area_pixels": {
            "min": min(areas, default=0),
            "max": max(areas, default=0),
            "mean": round(float(np.mean(areas)), 6) if areas else 0.0,
            "median": round(float(np.median(areas)), 6) if areas else 0.0,
        },
        "manual_review_candidates": len(review_candidates(image_records)),
        "review_reason_counts": dict(sorted(Counter(
            reason for record in image_records for reason in record["manual_review_reasons"].split("|") if reason
        ).items())),
    }


def write_montage(path: Path, dataset_root: Path, candidates: list[dict[str, Any]], *, sample_size: int) -> None:
    """Render a small deterministic candidate montage for manual protocol development."""
    selected = candidates[:sample_size]
    if not selected:
        return
    columns = 3
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, record in zip(axes.flat, selected):
        with Image.open(Path(dataset_root) / record["image_path"]) as image_file:
            axis.imshow(image_file.convert("RGB"))
        axis.set_title(
            f"{record['stable_source_image_id']}\npriority={record['manual_review_priority']} {record['manual_review_reasons']}",
            fontsize=7,
        )
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tiny-area-pixels", type=int, default=64)
    parser.add_argument("--large-area-pixels", type=int, default=50_000)
    parser.add_argument("--unusual-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--montage-size", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiny_area_pixels < 1 or args.large_area_pixels < args.tiny_area_pixels or args.unusual_aspect_ratio <= 1:
        raise ValueError("Component-audit thresholds must be positive and internally consistent.")
    splits = require_development_splits(args.splits)
    rows = load_development_manifest_rows(
        args.manifest_root,
        {"train": args.train_manifest, "val": args.val_manifest},
        splits,
    )
    components, images = audit_records(
        args.dataset_root,
        rows,
        tiny_area_pixels=args.tiny_area_pixels,
        large_area_pixels=args.large_area_pixels,
        unusual_aspect_ratio=args.unusual_aspect_ratio,
    )
    candidates = review_candidates(images)
    summary = summarize_audit(components, images, splits)
    write_csv(args.output_dir / "big_rock_component_candidates.csv", components, COMPONENT_FIELDS)
    write_csv(args.output_dir / "big_rock_component_images.csv", images, IMAGE_FIELDS)
    write_csv(args.output_dir / "manual_review_candidates.csv", candidates, IMAGE_FIELDS)
    write_json(args.output_dir / "big_rock_component_audit_summary.json", summary)
    write_montage(args.output_dir / "manual_review_montage.png", args.dataset_root, candidates, sample_size=args.montage_size)
    print(summary)


if __name__ == "__main__":
    main()