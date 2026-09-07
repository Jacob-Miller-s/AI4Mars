import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from PIL import Image

from src.rock_instance.annotations import (
    REVIEW_VERSION,
    TERMINAL_ANNOTATION_STATUSES,
    configure_component_review,
    configure_review_scope,
    initial_calibration_reference,
    initialize_review_state,
    load_review_state,
    record_annotation,
    save_review_state,
    unresolved_candidate_component_ids,
)
from src.rock_instance.review_tool import RockInstanceReviewUI


class RockInstanceReviewUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "dataset"
        self.image_relative = "msl/ncam/images/edr/source-a.JPG"
        self.mask_relative = "msl/ncam/labels/train/source-a.png"
        image_path = self.dataset_root / self.image_relative
        mask_path = self.dataset_root / self.mask_relative
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        Image.new("RGB", (12, 10), color=(10, 20, 30)).save(image_path)
        Image.new("L", (12, 10), color=0).save(mask_path)

        self.candidate_manifest = self.root / "candidates.csv"
        with self.candidate_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pilot_rank", "stable_source_image_id", "split", "sequence_id", "image_path", "mask_path",
                    "annotation_status", "selection_strata",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "pilot_rank": 1,
                    "stable_source_image_id": "source-a",
                    "split": "train",
                    "sequence_id": "sequence-a",
                    "image_path": self.image_relative,
                    "mask_path": self.mask_relative,
                    "annotation_status": "candidate_unreviewed",
                    "selection_strata": "multiple_candidate_regions",
                }
            )
        self.component_manifest = self.root / "components.csv"
        self.component_manifest.write_text(
            "stable_source_image_id,component_id,bbox_left,bbox_top,bbox_width,bbox_height\n"
            "source-a,1,1,1,3,3\nsource-a,2,6,1,3,3\n",
            encoding="utf-8",
        )
        self.calibration_manifest = self.root / "calibration.csv"
        self.calibration_manifest.write_text("stable_source_image_id\nsource-a\n", encoding="utf-8")
        self.protocol_path = self.root / "annotation_protocol_v2.md"
        self.protocol_path.write_text("# v2.0-calibration-resolved\n", encoding="utf-8")

        initial_state = initialize_review_state(self.candidate_manifest, self.dataset_root)
        record_annotation(
            initial_state,
            {
                "instance_id": "initial:source-a",
                "image_id": "source-a",
                "sequence_id": "sequence-a",
                "source_candidate_component_id": 1,
                "bbox": [1, 1, 3, 3],
                "annotation_status": "rejected_noise",
                "discrete_rock": False,
                "truncated": False,
                "occluded": False,
                "uncertain": False,
                "reviewer_notes": "initial review",
                "review_version": REVIEW_VERSION,
            },
            reviewer="initial_reviewer",
        )
        snapshot_path = self.root / "review_state_initial_v1.json"
        save_review_state(snapshot_path, initial_state)

        corrected_state = initialize_review_state(self.candidate_manifest, self.dataset_root, pilot_id="calibration_resolved_v2")
        configure_review_scope(
            corrected_state,
            name="calibration",
            image_ids=["source-a"],
            source_manifest=self.calibration_manifest,
        )
        configure_component_review(
            corrected_state,
            component_manifest=self.component_manifest,
            component_ids_by_image={"source-a": [1, 2]},
            protocol_path=self.protocol_path,
            initial_calibration_reference=initial_calibration_reference(snapshot_path),
        )
        self.state_path = self.root / "review_state.json"
        save_review_state(self.state_path, corrected_state)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_multi_source_save_creates_a_merge_resolution_and_covers_every_source(self) -> None:
        ui = RockInstanceReviewUI(
            state_path=self.state_path,
            component_candidates_csv=self.component_manifest,
            dataset_root=self.dataset_root,
            image_id="source-a",
            reviewer="researcher",
        )
        try:
            self.assertEqual({label.get_text() for label in ui.status_control.labels}, TERMINAL_ANNOTATION_STATUSES)
            ui.merge_mode = True
            ui.selected_source_component_ids = {1, 2}
            ui.notes = "The separated semantic regions are one physical rock."
            ui._save_decision(None)
        finally:
            matplotlib.pyplot.close(ui.figure)

        saved_state = load_review_state(self.state_path)
        saved_annotation = saved_state["images"]["source-a"]["annotations"][0]
        self.assertEqual(saved_annotation["annotation_status"], "uncertain")
        self.assertEqual(saved_annotation["source_candidate_component_ids"], [1, 2])
        self.assertEqual(saved_state["resolution_records"][0]["resolution_type"], "merge")
        self.assertEqual(unresolved_candidate_component_ids(saved_state, "source-a"), [])

    def test_direct_save_uses_the_highlighted_tiny_component(self) -> None:
        ui = RockInstanceReviewUI(
            state_path=self.state_path,
            component_candidates_csv=self.component_manifest,
            dataset_root=self.dataset_root,
            image_id="source-a",
            reviewer="researcher",
        )
        try:
            ui._cycle_component(1)
            ui.status = "rejected_noise"
            ui._save_decision(None)
        finally:
            matplotlib.pyplot.close(ui.figure)

        saved_state = load_review_state(self.state_path)
        saved_annotation = saved_state["images"]["source-a"]["annotations"][0]
        self.assertEqual(saved_annotation["annotation_status"], "rejected_noise")
        self.assertEqual(saved_annotation["source_candidate_component_ids"], [2])
        self.assertEqual(unresolved_candidate_component_ids(saved_state, "source-a"), [1])

    def test_three_review_panel_clicks_are_saved_as_an_accepted_polygon(self) -> None:
        ui = RockInstanceReviewUI(
            state_path=self.state_path,
            component_candidates_csv=self.component_manifest,
            dataset_root=self.dataset_root,
            image_id="source-a",
            reviewer="researcher",
        )
        try:
            ui._on_status_changed("accepted")
            ui._toggle_draw_mode(None)
            for x_coordinate, y_coordinate in ((1.0, 1.0), (4.0, 1.0), (2.0, 4.0)):
                ui._on_canvas_click(
                    SimpleNamespace(inaxes=ui.review_axis, xdata=x_coordinate, ydata=y_coordinate)
                )
            self.assertEqual(len(ui.polygon), 3)
            ui._save_decision(None)
        finally:
            matplotlib.pyplot.close(ui.figure)

        saved_state = load_review_state(self.state_path)
        saved_annotation = saved_state["images"]["source-a"]["annotations"][0]
        self.assertEqual(saved_annotation["annotation_status"], "accepted")
        self.assertEqual(saved_annotation["polygon"], [[1.0, 1.0], [4.0, 1.0], [2.0, 4.0]])

    def test_boundary_indeterminate_save_preserves_identity_without_polygon(self) -> None:
        ui = RockInstanceReviewUI(
            state_path=self.state_path,
            component_candidates_csv=self.component_manifest,
            dataset_root=self.dataset_root,
            image_id="source-a",
            reviewer="researcher",
        )
        try:
            ui._on_status_changed("boundary_indeterminate")
            ui.notes = "The physical rock is present, but RGB does not support a reproducible visible boundary."
            ui._save_decision(None)
        finally:
            matplotlib.pyplot.close(ui.figure)

        saved_state = load_review_state(self.state_path)
        saved_annotation = saved_state["images"]["source-a"]["annotations"][0]
        self.assertEqual(saved_annotation["annotation_status"], "boundary_indeterminate")
        self.assertTrue(saved_annotation["discrete_rock"])
        self.assertIsNone(saved_annotation["polygon"])
        self.assertEqual(unresolved_candidate_component_ids(saved_state, "source-a"), [2])

    def test_accepted_merge_uses_one_polygon_from_the_selected_source(self) -> None:
        ui = RockInstanceReviewUI(
            state_path=self.state_path,
            component_candidates_csv=self.component_manifest,
            dataset_root=self.dataset_root,
            image_id="source-a",
            reviewer="researcher",
        )
        try:
            ui.merge_mode = True
            ui.selected_component_id = 2
            ui.selected_source_component_ids = {1, 2}
            ui.notes = "Both semantic regions are one physical rock."
            ui._on_status_changed("accepted")
            ui._toggle_draw_mode(None)
            for x_coordinate, y_coordinate in ((6.0, 1.0), (9.0, 1.0), (7.0, 4.0)):
                ui._on_canvas_click(
                    SimpleNamespace(inaxes=ui.review_axis, xdata=x_coordinate, ydata=y_coordinate)
                )
            ui._save_decision(None)
        finally:
            matplotlib.pyplot.close(ui.figure)

        saved_state = load_review_state(self.state_path)
        saved_annotation = saved_state["images"]["source-a"]["annotations"][0]
        self.assertEqual(saved_annotation["source_candidate_component_id"], 2)
        self.assertEqual(saved_annotation["source_candidate_component_ids"], [1, 2])
        self.assertEqual(len(saved_annotation["polygon"]), 3)
        self.assertEqual(saved_state["resolution_records"][0]["source_candidate_component_ids"], [1, 2])


if __name__ == "__main__":
    unittest.main()