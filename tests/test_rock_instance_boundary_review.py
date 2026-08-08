import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rock_instance.boundary_review import (
    BOUNDARY_REVIEW_SCHEMA_VERSION,
    BOUNDARY_REVIEW_VERSION,
    _accepted_annotations_for_component,
    _forensic_filename,
    activate_interactive_backend,
    finish_boundary_target,
    load_boundary_review_state,
    record_boundary_redraw,
    record_identity_escalation,
    save_boundary_review_state,
    validate_boundary_review_state,
)


def _target(target_id: str) -> dict:
    return {
        "target_id": target_id, "image_id": "image-a", "sequence_id": "sequence-a", "image_path": "image.JPG", "mask_path": "mask.png",
        "image_width": 20, "image_height": 20, "source_candidate_component_id": 1, "v21_instance_id": "image-a:rock-001",
        "object_identity_fixed": "accepted", "boundary_question": "What visible pixels belong?", "review_status": "unreviewed", "reviewer": None,
        "polygon": None, "bbox": None, "reviewer_notes": "", "identity_escalation": False, "identity_escalation_note": "",
    }


def _state() -> dict:
    targets = [_target(f"target-{index}") for index in range(1, 4)]
    return {
        "schema_version": BOUNDARY_REVIEW_SCHEMA_VERSION, "review_version": BOUNDARY_REVIEW_VERSION, "expert_splits_excluded": True,
        "review_scope": {"name": "boundary_clarification", "target_ids": [target["target_id"] for target in targets], "source_manifest": "targets.csv", "source_manifest_sha256": "a" * 64},
        "provenance": {"primary_state_sha256": "a" * 64, "repeat_state_sha256": "b" * 64, "v21_state_sha256": "c" * 64, "proposed_protocol_sha256": "d" * 64, "component_manifest_sha256": "e" * 64, "historic_annotations_hidden": True, "v21_polygons_hidden": True},
        "component_candidates_csv": "components.csv", "proposed_protocol": {"version": BOUNDARY_REVIEW_VERSION, "path": "protocol.md", "sha256": "d" * 64}, "targets": targets,
    }


class BoundaryReviewTests(unittest.TestCase):
    def test_interactive_mode_selects_tk_backend_before_creating_ui(self) -> None:
        with patch("src.rock_instance.boundary_review.plt.switch_backend") as switch_backend:
            activate_interactive_backend()
        switch_backend.assert_called_once_with("TkAgg")

    def test_forensic_filename_is_windows_safe(self) -> None:
        self.assertEqual(
            _forensic_filename("NLB_463551084EDR_F0411534NCAM00385M1:component-4"),
            "NLB_463551084EDR_F0411534NCAM00385M1__component-4.png",
        )

    def test_forensic_selection_recognizes_scalar_component_provenance(self) -> None:
        record = {
            "annotations": [
                {"annotation_status": "accepted", "source_candidate_component_id": 7},
                {"annotation_status": "accepted", "source_candidate_component_ids": [8]},
                {"annotation_status": "rejected_noise", "source_candidate_component_id": 7},
            ]
        }
        selected = _accepted_annotations_for_component(record, 7)
        self.assertEqual(selected, [record["annotations"][0]])

    def test_in_progress_redraw_can_be_saved_and_reloaded(self) -> None:
        state = _state()
        record_boundary_redraw(state, "target-1", polygon=[[1, 1], [5, 1], [3, 5]], reviewer="reviewer", notes="visible edge")
        validate_boundary_review_state(state)
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "review_state.json"
            save_boundary_review_state(state_path, state)
            reloaded = load_boundary_review_state(state_path)
        self.assertEqual(reloaded["targets"][0]["review_status"], "in_progress")
        self.assertEqual(reloaded["targets"][0]["polygon"], [[1, 1], [5, 1], [3, 5]])

    def test_redraw_and_identity_escalation_cannot_change_fixed_identity(self) -> None:
        state = _state()
        record_boundary_redraw(state, "target-1", polygon=[[1, 1], [5, 1], [3, 5]], reviewer="reviewer", notes="visible edge")
        finish_boundary_target(state, "target-1", reviewer="reviewer")
        record_identity_escalation(state, "target-2", reviewer="reviewer", note="proposal reference does not locate a defensible fixed object")
        finish_boundary_target(state, "target-2", reviewer="reviewer")
        validate_boundary_review_state(state)
        self.assertEqual(state["targets"][0]["review_status"], "redrawn")
        self.assertEqual(state["targets"][0]["object_identity_fixed"], "accepted")
        self.assertEqual(state["targets"][1]["review_status"], "identity_escalated")
        self.assertIsNone(state["targets"][1]["polygon"])


if __name__ == "__main__":
    unittest.main()