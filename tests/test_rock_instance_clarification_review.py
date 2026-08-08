import csv
import json
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
    record_annotation,
    save_review_state,
    sha256_file,
)
from src.rock_instance.clarification_review import prepare_clarification_review


class ClarificationReviewTests(unittest.TestCase):
    def test_prepared_clarification_state_is_empty_and_preserves_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            image_relative = "msl/ncam/images/edr/source-a.JPG"
            mask_relative = "msl/ncam/labels/train/source-a.png"
            (dataset_root / image_relative).parent.mkdir(parents=True)
            (dataset_root / mask_relative).parent.mkdir(parents=True)
            Image.new("RGB", (12, 10)).save(dataset_root / image_relative)
            Image.new("L", (12, 10)).save(dataset_root / mask_relative)
            candidates = root / "candidates.csv"
            with candidates.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pilot_rank", "stable_source_image_id", "split", "sequence_id", "image_path", "mask_path", "annotation_status", "selection_strata"])
                writer.writeheader()
                writer.writerow({"pilot_rank": 1, "stable_source_image_id": "source-a", "split": "train", "sequence_id": "sequence-a", "image_path": image_relative, "mask_path": mask_relative, "annotation_status": "candidate_unreviewed", "selection_strata": "isolated_candidate"})
            components = root / "components.csv"
            components.write_text("stable_source_image_id,component_id\nsource-a,1\n", encoding="utf-8")
            calibration = root / "calibration.csv"
            calibration.write_text("stable_source_image_id\nsource-a\n", encoding="utf-8")
            protocol = root / "protocol.md"
            protocol.write_text("# protocol\n", encoding="utf-8")
            annotation = {"instance_id": "source-a:rock-001", "image_id": "source-a", "sequence_id": "sequence-a", "source_candidate_component_id": 1, "bbox": [1, 1, 3, 3], "annotation_status": "rejected_noise", "discrete_rock": False, "truncated": False, "occluded": False, "uncertain": False, "reviewer_notes": "noise", "review_version": REVIEW_VERSION}
            initial = initialize_review_state(candidates, dataset_root)
            record_annotation(initial, annotation, reviewer="initial", image_review_status="reviewed")
            initial_path = root / "initial.json"
            save_review_state(initial_path, initial)
            primary = initialize_review_state(candidates, dataset_root, pilot_id="calibration_resolved_v2")
            configure_review_scope(primary, name="calibration", image_ids=["source-a"], source_manifest=calibration)
            configure_component_review(primary, component_manifest=components, component_ids_by_image={"source-a": [1]}, protocol_path=protocol, initial_calibration_reference=initial_calibration_reference(initial_path))
            record_annotation(primary, annotation, reviewer="primary", image_review_status="reviewed")
            primary_path = root / "primary.json"
            save_review_state(primary_path, primary)
            repeat = json.loads(json.dumps(primary))
            repeat["review_scope"]["name"] = "calibration_repeat"
            repeat["repeat_review"] = {"selection_version": "test", "selection_seed": 42, "source_protocol_version": "v2.0-calibration-resolved"}
            repeat_path = root / "repeat.json"
            save_review_state(repeat_path, repeat)
            agreement = root / "agreement.json"
            agreement.write_text(json.dumps({"analysis_type": "intra-rater consistency", "CALIBRATION_PROTOCOL_RECOMMENDATION": "CLARIFY", "provenance": {"primary_state_sha256": sha256_file(primary_path), "repeat_state_sha256": sha256_file(repeat_path)}}), encoding="utf-8")
            selection = root / "selection.csv"
            selection.write_text("clarification_rank,stable_source_image_id,inclusion_reason,human_question\n1,source-a,reason,question\n", encoding="utf-8")
            proposed = root / "proposed.md"
            proposed.write_text("# proposed\n", encoding="utf-8")
            primary_before = primary_path.read_bytes()
            repeat_before = repeat_path.read_bytes()

            state_path = prepare_clarification_review(primary_state_path=primary_path, repeat_state_path=repeat_path, agreement_report_path=agreement, component_candidates_csv=components, selection_manifest=selection, proposed_protocol_path=proposed, output_dir=root / "clarification")

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["review_scope"]["name"], "calibration_clarification")
            self.assertEqual(state["review_scope"]["image_ids"], ["source-a"])
            self.assertEqual(state["images"]["source-a"]["annotations"], [])
            self.assertEqual(state["resolution_records"], [])
            self.assertTrue(state["clarification_review"]["prior_decisions_hidden"])
            self.assertEqual(primary_path.read_bytes(), primary_before)
            self.assertEqual(repeat_path.read_bytes(), repeat_before)


if __name__ == "__main__":
    unittest.main()