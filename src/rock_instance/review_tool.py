"""Resumable matplotlib-assisted review of Sprint 0 rock-component candidates."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from src.dataset import normalize_ai4mars_mask
from src.rock_instance.annotations import (
    ANNOTATION_STATUSES,
    REVIEW_VERSION,
    initialize_review_state,
    load_review_state,
    record_annotation,
    save_review_state,
    sha256_file,
)


NAV_CONTEXT_COLORS = np.array([[120, 86, 53], [115, 128, 132], [218, 179, 74], [201, 67, 45], [0, 0, 0]], dtype=np.uint8)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def initialize_pilot_artifacts(candidate_manifest: Path, dataset_root: Path, output_dir: Path) -> Path:
    """Preserve candidate provenance and initialize an empty, resumable reviewed-state file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_copy = output_dir / "rock_instance_pilot_candidates.csv"
    shutil.copyfile(candidate_manifest, candidate_copy)
    state = initialize_review_state(candidate_copy, dataset_root)
    state_path = output_dir / "review_state.json"
    save_review_state(state_path, state)
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": state["schema_version"],
                "review_version": state["review_version"],
                "candidate_manifest_sha256": sha256_file(candidate_copy),
                "candidate_manifest_source": str(Path(candidate_manifest)),
                "dataset_root": str(Path(dataset_root)),
                "expert_splits_excluded": True,
                "generation_module": "src.rock_instance.review_tool",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return state_path


def candidate_components(component_csv: Path, image_id: str) -> list[dict[str, str]]:
    """Return semantic candidate boxes for display and explicit reviewer references."""
    return [row for row in _read_csv(component_csv) if row["stable_source_image_id"] == image_id]


def next_unreviewed_image_id(state: dict[str, Any]) -> str:
    """Find the lowest-rank unreviewed/deferred image so sessions can be resumed."""
    available = [
        record for record in state["images"].values()
        if record["review_status"] in {"unreviewed", "deferred"}
    ]
    if not available:
        raise ValueError("All pilot images are reviewed.")
    return min(available, key=lambda record: record["pilot_rank"])["image_id"]


def render_review_image(
    state: dict[str, Any],
    image_id: str,
    dataset_root: Path,
    components: list[dict[str, str]],
    *,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Show RGB, terrain context, and semantic candidate components without writing source data."""
    image = state["images"].get(image_id)
    if image is None:
        raise ValueError(f"Unknown image_id: {image_id!r}")
    with Image.open(Path(dataset_root) / image["image_path"]) as image_file:
        rgb = np.asarray(image_file.convert("RGB"))
    mask_path = Path(dataset_root) / image["mask_path"]
    with Image.open(mask_path) as mask_file:
        mask = normalize_ai4mars_mask(np.asarray(mask_file, dtype=np.int64), mask_path)
    color_indices = np.where((mask >= 0) & (mask <= 3), mask, 4)
    context = NAV_CONTEXT_COLORS[color_indices]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB: {image_id}")
    axes[1].imshow(context)
    axes[1].set_title("NAV context: soil / bedrock / sand / Big Rock")
    axes[2].imshow(rgb)
    axes[2].imshow(np.ma.masked_where(mask != 3, mask), cmap="Reds", alpha=0.45, vmin=0, vmax=3)
    for component in components:
        rectangle = Rectangle(
            (int(component["bbox_left"]), int(component["bbox_top"])),
            int(component["bbox_width"]),
            int(component["bbox_height"]),
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
        )
        axes[2].add_patch(rectangle)
        axes[2].text(rectangle.get_x(), rectangle.get_y(), component["component_id"], color="cyan", fontsize=8)
    axes[2].set_title("Big Rock semantic candidates; IDs are review references only")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def _component_bbox(components: list[dict[str, str]], component_id: int) -> list[int]:
    for component in components:
        if int(component["component_id"]) == component_id:
            return [int(component["bbox_left"]), int(component["bbox_top"]), int(component["bbox_width"]), int(component["bbox_height"])]
    raise ValueError(f"Component ID {component_id} is not available for this pilot image.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true", help="Create candidate copy, provenance, and empty review state.")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--component-candidates-csv", type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-preview", type=Path)
    parser.add_argument("--action", choices=sorted(ANNOTATION_STATUSES))
    parser.add_argument("--instance-id")
    parser.add_argument("--component-id", type=int)
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--polygon-json", help="JSON list of reviewer-drawn [x, y] polygon points; required for accepted rocks.")
    parser.add_argument("--truncated", action="store_true")
    parser.add_argument("--occluded", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--reviewer", default="single_researcher")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initialize:
        if args.candidate_manifest is None or args.output_dir is None:
            raise ValueError("--initialize requires --candidate-manifest and --output-dir.")
        print(initialize_pilot_artifacts(args.candidate_manifest, args.dataset_root, args.output_dir))
        return
    if args.state_path is None or args.component_candidates_csv is None:
        raise ValueError("Review actions require --state-path and --component-candidates-csv.")
    state = load_review_state(args.state_path)
    image_id = args.image_id or next_unreviewed_image_id(state)
    components = candidate_components(args.component_candidates_csv, image_id)
    if args.show or args.save_preview:
        render_review_image(state, image_id, args.dataset_root, components, output_path=args.save_preview, show=args.show)
    if args.action is None:
        print(image_id)
        return
    if args.instance_id is None:
        raise ValueError("--action requires --instance-id.")
    bbox = list(args.bbox) if args.bbox is not None else None
    if bbox is None and args.component_id is not None:
        bbox = _component_bbox(components, args.component_id)
    if bbox is None:
        raise ValueError("Supply --bbox or --component-id to identify the reviewed region.")
    image = state["images"][image_id]
    annotation = {
        "instance_id": args.instance_id,
        "image_id": image_id,
        "sequence_id": image["sequence_id"],
        "source_candidate_component_id": args.component_id,
        "bbox": bbox,
        "polygon": json.loads(args.polygon_json) if args.polygon_json else None,
        "annotation_status": args.action,
        "discrete_rock": args.action == "accepted",
        "truncated": args.truncated,
        "occluded": args.occluded,
        "uncertain": args.action == "uncertain",
        "reviewer_notes": args.notes,
        "review_version": REVIEW_VERSION,
    }
    record_annotation(state, annotation, reviewer=args.reviewer)
    save_review_state(args.state_path, state)
    print(f"saved {args.action} decision for {image_id}")


if __name__ == "__main__":
    main()