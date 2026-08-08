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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons
from PIL import Image

from src.dataset import normalize_ai4mars_mask
from src.rock_instance.annotations import (
    ANNOTATION_STATUSES,
    REVIEW_VERSION,
    configure_component_review,
    configure_review_scope,
    finish_image_review,
    initial_calibration_reference,
    initialize_review_state,
    load_review_state,
    record_annotation,
    record_resolution,
    save_review_state,
    set_candidate_independent_observation,
    sha256_file,
)


NAV_CONTEXT_COLORS = np.array([[120, 86, 53], [115, 128, 132], [218, 179, 74], [201, 67, 45], [0, 0, 0]], dtype=np.uint8)
MAX_DISPLAY_DIMENSION = 640


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


def activate_calibration_scope(state_path: Path, calibration_manifest: Path) -> None:
    """Restrict an empty pilot state to its deterministic calibration subset."""
    state_path = Path(state_path)
    state = load_review_state(state_path)
    calibration_manifest = Path(calibration_manifest)
    calibration_rows = _read_csv(calibration_manifest)
    calibration_ids = [row["stable_source_image_id"] for row in calibration_rows]
    configure_review_scope(
        state,
        name="calibration",
        image_ids=calibration_ids,
        source_manifest=calibration_manifest,
    )
    calibration_copy = state_path.parent / calibration_manifest.name
    if calibration_copy.exists() and sha256_file(calibration_copy) != sha256_file(calibration_manifest):
        raise FileExistsError(f"Refusing to replace a different calibration manifest: {calibration_copy}")
    if not calibration_copy.exists():
        shutil.copyfile(calibration_manifest, calibration_copy)
    save_review_state(state_path, state)


def initialize_calibration_resolution_artifacts(
    *,
    candidate_manifest: Path,
    component_candidates_csv: Path,
    calibration_manifest: Path,
    initial_snapshot: Path,
    protocol_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> Path:
    """Create a fresh v2 calibration state without changing prior human-review evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "review_state.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing corrected calibration state: {state_path}")
    copied_paths: dict[str, Path] = {}
    for source_path, destination_name in (
        (candidate_manifest, "rock_instance_pilot_candidates.csv"),
        (component_candidates_csv, "big_rock_component_candidates.csv"),
        (calibration_manifest, "rock_instance_pilot_calibration_candidates.csv"),
        (protocol_path, "annotation_protocol_v2.0-calibration-resolved.md"),
    ):
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Required calibration-resolution source is missing: {source_path}")
        destination_path = output_dir / destination_name
        shutil.copyfile(source_path, destination_path)
        copied_paths[destination_name] = destination_path
    calibration_rows = _read_csv(copied_paths["rock_instance_pilot_calibration_candidates.csv"])
    calibration_ids = [row["stable_source_image_id"] for row in calibration_rows]
    component_ids_by_image: dict[str, list[int]] = {image_id: [] for image_id in calibration_ids}
    for component in _read_csv(copied_paths["big_rock_component_candidates.csv"]):
        image_id = component["stable_source_image_id"]
        if image_id in component_ids_by_image:
            component_ids_by_image[image_id].append(int(component["component_id"]))
    state = initialize_review_state(copied_paths["rock_instance_pilot_candidates.csv"], dataset_root, pilot_id="calibration_resolved_v2")
    configure_review_scope(
        state,
        name="calibration",
        image_ids=calibration_ids,
        source_manifest=copied_paths["rock_instance_pilot_calibration_candidates.csv"],
    )
    configure_component_review(
        state,
        component_manifest=copied_paths["big_rock_component_candidates.csv"],
        component_ids_by_image=component_ids_by_image,
        protocol_path=copied_paths["annotation_protocol_v2.0-calibration-resolved.md"],
        initial_calibration_reference=initial_calibration_reference(initial_snapshot),
    )
    save_review_state(state_path, state)
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "calibration_resolution_version": "v2.0-calibration-resolved",
                "initial_snapshot_path": str(Path(initial_snapshot)),
                "initial_snapshot_sha256": sha256_file(initial_snapshot),
                "candidate_manifest_source": str(Path(candidate_manifest)),
                "component_candidates_source": str(Path(component_candidates_csv)),
                "calibration_manifest_source": str(Path(calibration_manifest)),
                "protocol_source": str(Path(protocol_path)),
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
    scope_image_ids = state.get("review_scope", {}).get("image_ids")
    scope_ranks = {image_id: rank for rank, image_id in enumerate(scope_image_ids or [])}
    available = [
        record for record in state["images"].values()
        if record["review_status"] in {"unreviewed", "in_progress", "deferred"}
        and (scope_image_ids is None or record["image_id"] in scope_ranks)
    ]
    if not available:
        scope_name = state.get("review_scope", {}).get("name", "pilot")
        raise ValueError(f"All images in the active {scope_name} review scope are reviewed.")
    return min(
        available,
        key=lambda record: scope_ranks.get(record["image_id"], record["pilot_rank"]),
    )["image_id"]


def _bbox_from_polygon(polygon: list[list[float]]) -> list[int]:
    """Compute a source-image bounding box that encloses a reviewer-drawn polygon."""
    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    left, top = int(np.floor(min(x_values))), int(np.floor(min(y_values)))
    right, bottom = int(np.floor(max(x_values))) + 1, int(np.floor(max(y_values))) + 1
    return [left, top, max(1, right - left), max(1, bottom - top)]


def _next_instance_id(state: dict[str, Any], image_id: str) -> str:
    """Generate a readable, unique instance ID without making an object decision."""
    existing_ids = {annotation["instance_id"] for image in state["images"].values() for annotation in image["annotations"]}
    index = 1
    while f"{image_id}:rock-{index:03d}" in existing_ids:
        index += 1
    return f"{image_id}:rock-{index:03d}"


def _downsample_for_display(array: np.ndarray, *, resample: Image.Resampling) -> np.ndarray:
    """Bound display work while keeping plot coordinates in source-image pixels."""
    height, width = array.shape[:2]
    scale = min(1.0, MAX_DISPLAY_DIMENSION / max(height, width))
    if scale == 1.0:
        return array
    display_size = (round(width * scale), round(height * scale))
    return np.asarray(Image.fromarray(array).resize(display_size, resample=resample))


class RockInstanceReviewUI:
    """Human-operated, atomic-save review window for one bounded pilot queue."""

    def __init__(
        self,
        *,
        state_path: Path,
        component_candidates_csv: Path,
        dataset_root: Path,
        image_id: str | None,
        reviewer: str,
    ) -> None:
        self.state_path = Path(state_path)
        self.component_candidates_csv = Path(component_candidates_csv)
        self.dataset_root = Path(dataset_root)
        self.reviewer = reviewer
        self.state = load_review_state(self.state_path)
        self.figure = plt.figure(figsize=(20, 11))
        self.figure.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._load_image(image_id or next_unreviewed_image_id(self.state))

    def _load_image(self, image_id: str) -> None:
        self.image_id = image_id
        self.image = self.state["images"][image_id]
        self.components = candidate_components(self.component_candidates_csv, image_id)
        if not self.components:
            raise ValueError(f"No semantic component candidates are available for {image_id}.")
        self.selected_component_id = int(self.components[0]["component_id"])
        self.status = "uncertain"
        self.polygon: list[list[float]] = []
        self.draw_mode = False
        self.notes = ""
        self._render()

    def _render(self) -> None:
        self._disconnect_controls()
        self.figure.clear()
        self.overlay_artists = []
        self.rgb_axis = self.figure.add_axes((0.02, 0.34, 0.30, 0.60))
        self.context_axis = self.figure.add_axes((0.35, 0.34, 0.30, 0.60))
        self.review_axis = self.figure.add_axes((0.68, 0.34, 0.30, 0.60))
        with Image.open(self.dataset_root / self.image["image_path"]) as image_file:
            rgb = np.asarray(image_file.convert("RGB"))
        mask_path = self.dataset_root / self.image["mask_path"]
        with Image.open(mask_path) as mask_file:
            mask = normalize_ai4mars_mask(np.asarray(mask_file, dtype=np.int64), mask_path)
        color_indices = np.where((mask >= 0) & (mask <= 3), mask, 4)
        image_width, image_height = self.image["image_width"], self.image["image_height"]
        extent = (0, image_width, image_height, 0)
        display_rgb = _downsample_for_display(rgb, resample=Image.Resampling.BILINEAR)
        display_context = _downsample_for_display(NAV_CONTEXT_COLORS[color_indices], resample=Image.Resampling.NEAREST)
        display_big_rock = _downsample_for_display((mask == 3).astype(np.uint8), resample=Image.Resampling.NEAREST)
        self.rgb_axis.imshow(display_rgb, extent=extent)
        self.rgb_axis.set_title(f"RGB: {self.image_id}")
        self.context_axis.imshow(display_context, extent=extent)
        self.context_axis.set_title("NAV terrain context")
        self.review_axis.imshow(display_rgb, extent=extent)
        self.review_axis.imshow(
            np.ma.masked_where(display_big_rock == 0, display_big_rock),
            cmap="Reds",
            alpha=0.45,
            vmin=0,
            vmax=1,
            extent=extent,
        )
        self.review_axis.set_title("Human rock-instance review")
        for axis in (self.rgb_axis, self.context_axis, self.review_axis):
            axis.axis("off")
        self._draw_review_overlays()
        scope_name = self.state.get("review_scope", {}).get("name", "pilot")
        self.figure.suptitle(
            f"{scope_name} review | {self.image_id} | {len(self.components)} semantic candidates | object class: rock",
            fontsize=14,
        )
        self._build_controls()
        self.figure.canvas.draw_idle()

    def _disconnect_controls(self) -> None:
        for control in getattr(self, "controls", []):
            control.disconnect_events()
        self.controls: list[Any] = []

    def _draw_review_overlays(self) -> None:
        for artist in getattr(self, "overlay_artists", []):
            artist.remove()
        self.overlay_artists: list[Any] = []
        for component in self.components:
            component_id = int(component["component_id"])
            selected = component_id == self.selected_component_id
            rectangle = Rectangle(
                (int(component["bbox_left"]), int(component["bbox_top"])),
                int(component["bbox_width"]),
                int(component["bbox_height"]),
                linewidth=2.5 if selected else 1.5,
                edgecolor="yellow" if selected else "cyan",
                facecolor="none",
            )
            self.review_axis.add_patch(rectangle)
            label = self.review_axis.text(
                rectangle.get_x(), rectangle.get_y(), str(component_id), color="yellow" if selected else "cyan", fontsize=9
            )
            self.overlay_artists.extend((rectangle, label))
        for annotation in self.image["annotations"]:
            if annotation["annotation_status"] != "accepted":
                continue
            polygon = annotation["polygon"] + [annotation["polygon"][0]]
            line = Line2D(
                [point[0] for point in polygon], [point[1] for point in polygon], color="lime", linewidth=2.0
            )
            self.review_axis.add_line(line)
            self.overlay_artists.append(line)
        if self.polygon:
            polygon = self.polygon + ([self.polygon[0]] if len(self.polygon) > 2 else [])
            line = Line2D(
                [point[0] for point in polygon], [point[1] for point in polygon], color="white", linewidth=2.0, marker="o"
            )
            self.review_axis.add_line(line)
            self.overlay_artists.append(line)

    def _build_controls(self) -> None:
        status_axis = self.figure.add_axes((0.02, 0.035, 0.17, 0.25))
        self.status_control = RadioButtons(status_axis, sorted(ANNOTATION_STATUSES), active=sorted(ANNOTATION_STATUSES).index(self.status))
        self.status_control.on_clicked(self._on_status_changed)
        component_axis = self.figure.add_axes((0.23, 0.20, 0.17, 0.06))
        component_axis.axis("off")
        self.component_text = component_axis.text(0, 0.75, "", va="top", fontsize=10)
        previous_button = Button(self.figure.add_axes((0.41, 0.21, 0.10, 0.05)), "Previous")
        previous_button.on_clicked(lambda _event: self._cycle_component(-1))
        next_button = Button(self.figure.add_axes((0.53, 0.21, 0.10, 0.05)), "Next")
        next_button.on_clicked(lambda _event: self._cycle_component(1))
        notes_button = Button(self.figure.add_axes((0.65, 0.21, 0.12, 0.05)), "Edit notes")
        notes_button.on_clicked(self._edit_notes)
        flags_axis = self.figure.add_axes((0.23, 0.025, 0.15, 0.055))
        self.flags_control = CheckButtons(flags_axis, ("truncated", "occluded"), (False, False))
        self.draw_button = Button(self.figure.add_axes((0.79, 0.21, 0.17, 0.05)), "Draw polygon")
        self.draw_button.on_clicked(self._toggle_draw_mode)
        undo_button = Button(self.figure.add_axes((0.84, 0.21, 0.12, 0.05)), "Undo point")
        undo_button.on_clicked(self._undo_point)
        save_button = Button(self.figure.add_axes((0.40, 0.025, 0.17, 0.055)), "Save decision")
        save_button.on_clicked(self._save_decision)
        finish_button = Button(self.figure.add_axes((0.59, 0.025, 0.17, 0.055)), "Finish image")
        finish_button.on_clicked(self._finish_image)
        self.message_axis = self.figure.add_axes((0.78, 0.02, 0.18, 0.07))
        self.message_axis.axis("off")
        self.message_text = self.message_axis.text(0, 0.9, "", va="top", wrap=True, fontsize=10)
        self.controls = [
            self.status_control, previous_button, next_button, notes_button, self.flags_control,
            self.draw_button, undo_button, save_button, finish_button,
        ]
        self._update_component_text()

    def _set_message(self, message: str, *, error: bool = False) -> None:
        self.message_text.set_text(message)
        self.message_text.set_color("crimson" if error else "black")
        self.figure.canvas.draw_idle()

    def _on_status_changed(self, status: str) -> None:
        self.status = status
        if status != "accepted":
            self.draw_mode = False
            self.draw_button.label.set_text("Draw polygon")

    def _cycle_component(self, direction: int) -> None:
        component_ids = [int(component["component_id"]) for component in self.components]
        current_index = component_ids.index(self.selected_component_id)
        self.selected_component_id = component_ids[(current_index + direction) % len(component_ids)]
        self._update_component_text()
        self._draw_review_overlays()
        self.figure.canvas.draw_idle()

    def _update_component_text(self) -> None:
        self.component_text.set_text(f"Component: {self.selected_component_id}\nNotes: {self.notes or 'none'}")

    def _edit_notes(self, _event: Any) -> None:
        try:
            import tkinter as tk
            from tkinter import simpledialog

            parent = getattr(self.figure.canvas.manager, "window", None)
            root = None
            if not isinstance(parent, tk.Misc):
                root = tk.Tk()
                root.withdraw()
                parent = root
            notes = simpledialog.askstring("Rock instance review", "Reviewer notes", initialvalue=self.notes, parent=parent)
            if root is not None:
                root.destroy()
        except Exception as error:
            self._set_message(f"Unable to open notes dialog: {error}", error=True)
            return
        if notes is not None:
            self.notes = notes
            self._update_component_text()
            self.figure.canvas.draw_idle()

    def _toggle_draw_mode(self, _event: Any) -> None:
        if self.status != "accepted":
            self._set_message("Select accepted before drawing a visible-rock boundary.", error=True)
            return
        self.draw_mode = not self.draw_mode
        self.draw_button.label.set_text("Stop drawing" if self.draw_mode else "Draw polygon")
        self._set_message("Polygon drawing active." if self.draw_mode else "Polygon drawing paused.")
        self.figure.canvas.draw_idle()

    def _undo_point(self, _event: Any) -> None:
        if self.polygon:
            self.polygon.pop()
            self._draw_review_overlays()
            self.figure.canvas.draw_idle()

    def _on_canvas_click(self, event: Any) -> None:
        if event.inaxes is not self.review_axis or event.xdata is None or event.ydata is None:
            return
        if self.draw_mode:
            self.polygon.append([float(event.xdata), float(event.ydata)])
            self._draw_review_overlays()
            self.figure.canvas.draw_idle()
            return
        for component in self.components:
            left, top, width, height = _component_bbox(self.components, int(component["component_id"]))
            if left <= event.xdata <= left + width and top <= event.ydata <= top + height:
                self.selected_component_id = int(component["component_id"])
                self._update_component_text()
                self._draw_review_overlays()
                self.figure.canvas.draw_idle()
                return

    def _save_decision(self, _event: Any) -> None:
        try:
            component_id = self.selected_component_id
            bbox = _component_bbox(self.components, component_id)
            instance_id = _next_instance_id(self.state, self.image_id)
            if self.status == "accepted":
                if len(self.polygon) < 3:
                    raise ValueError("Accepted rocks require at least three reviewer-drawn polygon points.")
                bbox = _bbox_from_polygon(self.polygon)
            annotation = {
                "instance_id": instance_id,
                "image_id": self.image_id,
                "sequence_id": self.image["sequence_id"],
                "source_candidate_component_id": component_id,
                "bbox": bbox,
                "polygon": self.polygon if self.status == "accepted" else None,
                "annotation_status": self.status,
                "discrete_rock": self.status == "accepted",
                "truncated": self.flags_control.get_status()[0],
                "occluded": self.flags_control.get_status()[1],
                "uncertain": self.status == "uncertain",
                "reviewer_notes": self.notes,
                "review_version": REVIEW_VERSION,
            }
            record_annotation(self.state, annotation, reviewer=self.reviewer, image_review_status="in_progress")
            save_review_state(self.state_path, self.state)
        except (ValueError, TypeError) as error:
            self._set_message(str(error), error=True)
            return
        self.polygon = []
        self.draw_mode = False
        self.draw_button.label.set_text("Draw polygon")
        self.notes = ""
        self._update_component_text()
        self._draw_review_overlays()
        self._set_message("Decision saved. Finish the image only after all components are adjudicated.")

    def _finish_image(self, _event: Any) -> None:
        try:
            finish_image_review(self.state, self.image_id, reviewer=self.reviewer)
            save_review_state(self.state_path, self.state)
            next_image_id = next_unreviewed_image_id(self.state)
        except ValueError as error:
            if "All images in the active" in str(error):
                self._set_message("The active review scope is complete.")
            else:
                self._set_message(str(error), error=True)
            return
        self._load_image(next_image_id)

    def show(self) -> None:
        """Run the reviewer until the human closes its window."""
        plt.show()


def run_interactive_review(
    *,
    state_path: Path,
    component_candidates_csv: Path,
    dataset_root: Path,
    image_id: str | None,
    reviewer: str,
) -> None:
    """Launch the local human-review UI; only Save and Finish mutate review state."""
    RockInstanceReviewUI(
        state_path=state_path,
        component_candidates_csv=component_candidates_csv,
        dataset_root=dataset_root,
        image_id=image_id,
        reviewer=reviewer,
    ).show()


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
    parser.add_argument("--initialize-calibration-resolution", action="store_true", help="Create a fresh v2 corrected-calibration state from immutable initial evidence.")
    parser.add_argument("--activate-calibration-scope", action="store_true", help="Restrict an empty pilot state to a copied calibration manifest.")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--initial-snapshot", type=Path)
    parser.add_argument("--protocol-path", type=Path)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--component-candidates-csv", type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Launch the local human-review window with atomic Save and Finish controls.")
    parser.add_argument("--save-preview", type=Path)
    parser.add_argument("--action", choices=sorted(ANNOTATION_STATUSES))
    parser.add_argument("--instance-id")
    parser.add_argument("--component-id", type=int)
    parser.add_argument("--component-ids", nargs="+", type=int, help="Plural component provenance for a manual merge or related resolution.")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--polygon-json", help="JSON list of reviewer-drawn [x, y] polygon points; required for accepted rocks.")
    parser.add_argument("--truncated", action="store_true")
    parser.add_argument("--occluded", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--resolution-json", type=Path, help="Reviewer-authored split/merge resolution record JSON; never generated automatically.")
    parser.add_argument("--candidate-independent-observation", action="store_true", help="Record an image-level obvious-rock observation without creating an instance.")
    parser.add_argument("--reviewer", default="single_researcher")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initialize:
        if args.candidate_manifest is None or args.output_dir is None:
            raise ValueError("--initialize requires --candidate-manifest and --output-dir.")
        print(initialize_pilot_artifacts(args.candidate_manifest, args.dataset_root, args.output_dir))
        return
    if args.initialize_calibration_resolution:
        required = {
            "--candidate-manifest": args.candidate_manifest,
            "--component-candidates-csv": args.component_candidates_csv,
            "--calibration-manifest": args.calibration_manifest,
            "--initial-snapshot": args.initial_snapshot,
            "--protocol-path": args.protocol_path,
            "--output-dir": args.output_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"--initialize-calibration-resolution requires {', '.join(missing)}.")
        print(
            initialize_calibration_resolution_artifacts(
                candidate_manifest=args.candidate_manifest,
                component_candidates_csv=args.component_candidates_csv,
                calibration_manifest=args.calibration_manifest,
                initial_snapshot=args.initial_snapshot,
                protocol_path=args.protocol_path,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
            )
        )
        return
    if args.activate_calibration_scope:
        if args.state_path is None or args.calibration_manifest is None:
            raise ValueError("--activate-calibration-scope requires --state-path and --calibration-manifest.")
        activate_calibration_scope(args.state_path, args.calibration_manifest)
        print(f"activated calibration scope for {args.state_path}")
        return
    if args.state_path is None or args.component_candidates_csv is None:
        raise ValueError("Review actions require --state-path and --component-candidates-csv.")
    state = load_review_state(args.state_path)
    image_id = args.image_id or next_unreviewed_image_id(state)
    components = candidate_components(args.component_candidates_csv, image_id)
    if args.resolution_json is not None:
        if args.action is not None or args.interactive or args.candidate_independent_observation:
            raise ValueError("--resolution-json cannot be combined with annotation or observation actions.")
        record_resolution(state, json.loads(args.resolution_json.read_text(encoding="utf-8")))
        save_review_state(args.state_path, state)
        print(f"saved resolution record for {image_id}")
        return
    if args.candidate_independent_observation:
        if args.action is not None or args.interactive:
            raise ValueError("--candidate-independent-observation cannot be combined with annotation actions.")
        set_candidate_independent_observation(state, image_id, observed=True, note=args.notes)
        save_review_state(args.state_path, state)
        print(f"saved candidate-independent observation for {image_id}")
        return
    if args.interactive:
        if args.action is not None or args.show or args.save_preview is not None:
            raise ValueError("--interactive cannot be combined with --action, --show, or --save-preview.")
        run_interactive_review(
            state_path=args.state_path,
            component_candidates_csv=args.component_candidates_csv,
            dataset_root=args.dataset_root,
            image_id=args.image_id,
            reviewer=args.reviewer,
        )
        return
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
    if args.component_ids is not None:
        annotation["source_candidate_component_ids"] = args.component_ids
    record_annotation(state, annotation, reviewer=args.reviewer)
    save_review_state(args.state_path, state)
    print(f"saved {args.action} decision for {image_id}")


if __name__ == "__main__":
    main()