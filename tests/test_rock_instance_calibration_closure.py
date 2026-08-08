import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.rock_instance.annotations import (
    REVIEW_VERSION,
    configure_component_review,
    configure_review_scope,
    initial_calibration_reference,
    initialize_review_state,
    load_review_state,
    record_annotation,
    save_review_state,
)
from src.rock_instance.calibration_closure import audit_calibration_closure, prepare_repeat_review


class CalibrationClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "dataset"
        image_relative = "msl/ncam/images/edr/source-a.JPG"
        mask_relative = "msl/ncam/labels/train/source-a.png"
        image_path = self.dataset_root / image_relative
        mask_path = self.dataset_root / mask_relative
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        Image.new("RGB", (12, 10)).save(image_path)
        Image.new("L", (12, 10)).save(mask_path)
        self.candidates = self.root / "candidates.csv"
        with self.candidates.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pilot_rank", "stable_source_image_id", "split", "sequence_id", "image_path", "mask_path", "annotation_status", "selection_strata"])
            writer.writeheader()
            writer.writerow({"pilot_rank": 1, "stable_source_image_id": "source-a", "split": "train", "sequence_id": "sequence-a", "image_path": image_relative, "mask_path": mask_relative, "annotation_status": "candidate_unreviewed", "selection_strata": "isolated_candidate"})
        self.component_manifest = self.root / "components.csv"
        self.component_manifest.write_text("stable_source_image_id,component_id\nsource-a,1\n", encoding="utf-8")
        calibration_manifest = self.root / "calibration.csv"
        calibration_manifest.write_text("stable_source_image_id\nsource-a\n", encoding="utf-8")
        protocol_path = self.root / "protocol.md"
        protocol_path.write_text("# protocol\n", encoding="utf-8")
        initial = initialize_review_state(self.candidates, self.dataset_root)
        record_annotation(initial, self._annotation("initial:source-a"), reviewer="initial")
        initial_path = self.root / "initial.json"
        save_review_state(initial_path, initial)
        self.state = initialize_review_state(self.candidates, self.dataset_root, pilot_id="calibration_resolved_v2")
        configure_review_scope(self.state, name="calibration", image_ids=["source-a"], source_manifest=calibration_manifest)
        configure_component_review(self.state, component_manifest=self.component_manifest, component_ids_by_image={"source-a": [1]}, protocol_path=protocol_path, initial_calibration_reference=initial_calibration_reference(initial_path))
        self.primary_path = self.root / "primary.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _annotation(self, instance_id: str) -> dict:
        return {"instance_id": instance_id, "image_id": "source-a", "sequence_id": "sequence-a", "source_candidate_component_id": 1, "bbox": [1, 1, 3, 3], "annotation_status": "rejected_noise", "discrete_rock": False, "truncated": False, "occluded": False, "uncertain": False, "reviewer_notes": "noise", "review_version": REVIEW_VERSION}

    def test_completed_calibration_creates_isolated_repeat_and_keeps_freeze_blocked(self) -> None:
        record_annotation(self.state, self._annotation("resolved:source-a"), reviewer="researcher", image_review_status="reviewed")
        save_review_state(self.primary_path, self.state)

        repeat_state_path, closure = prepare_repeat_review(primary_state_path=self.primary_path, component_candidates_csv=self.component_manifest, output_dir=self.root / "repeat", target_size=1)

        self.assertTrue(repeat_state_path.is_file())
        self.assertTrue(closure["primary_calibration"]["complete"])
        self.assertEqual(closure["protocol_freeze_gate"]["status"], "blocked")
        self.assertIn("isolated_repeat_review_pending", closure["protocol_freeze_gate"]["blocking_conditions"])

    def test_incomplete_calibration_cannot_prepare_repeat(self) -> None:
        save_review_state(self.primary_path, self.state)

        with self.assertRaisesRegex(ValueError, "Repeat review is blocked"):
            prepare_repeat_review(primary_state_path=self.primary_path, component_candidates_csv=self.component_manifest, output_dir=self.root / "repeat", target_size=1)

        closure = audit_calibration_closure(self.state)
        self.assertFalse(closure["primary_calibration"]["complete"])
        self.assertIn("primary_calibration_incomplete", closure["protocol_freeze_gate"]["blocking_conditions"])

    def test_completed_repeat_with_clarify_report_keeps_gate_blocked(self) -> None:
        record_annotation(self.state, self._annotation("resolved:source-a"), reviewer="researcher", image_review_status="reviewed")
        save_review_state(self.primary_path, self.state)
        repeat_path, _ = prepare_repeat_review(
            primary_state_path=self.primary_path,
            component_candidates_csv=self.component_manifest,
            output_dir=self.root / "repeat",
            target_size=1,
        )
        repeat_state = load_review_state(repeat_path)
        record_annotation(repeat_state, self._annotation("repeat:source-a"), reviewer="researcher", image_review_status="reviewed")

        closure = audit_calibration_closure(
            self.state,
            repeat_state=repeat_state,
            agreement_report={"analysis_type": "intra-rater consistency", "CALIBRATION_PROTOCOL_RECOMMENDATION": "CLARIFY"},
        )

        self.assertEqual(closure["repeat_review"]["agreement_analysis"]["status"], "complete")
        self.assertIn("protocol_clarification_pending", closure["protocol_freeze_gate"]["blocking_conditions"])


if __name__ == "__main__":
    unittest.main()