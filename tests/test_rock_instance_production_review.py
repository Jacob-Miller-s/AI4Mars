import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.rock_instance.annotations import (
    BOUNDARY_INDETERMINATE_STATUS,
    REVIEW_VERSION,
    finish_image_review,
    load_review_state,
    maskrcnn_target_for_image,
    record_annotation,
    sha256_file,
)
from src.rock_instance.production_review import (
    CALIBRATION_SIZE,
    PRODUCTION_SIZE,
    PROTOCOL_FREEZE_STATUS,
    freeze_v23_protocol,
    prepare_production_review,
    summarize_production_review,
)


class ProductionReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "dataset"
        self.pilot_manifest = self.root / "pilot.csv"
        self.calibration_manifest = self.root / "calibration.csv"
        self.component_manifest = self.root / "components.csv"
        self.protocol_path = self.root / "annotation_protocol_v2.3-calibration-final.md"
        self.protocol_path.write_text("# v2.3-calibration-final\n", encoding="utf-8")
        self._write_manifests()
        self.closure_path = self.root / "closure.json"
        self.ledger_path = self.root / "ledger.json"
        self.calibration_state_path = self.root / "calibration-state.json"
        self.calibration_state_path.write_text("{}\n", encoding="utf-8")
        self.ledger_path.write_text(json.dumps({"schema_version": "rock_instance_calibration_finalization_v1"}), encoding="utf-8")
        self.closure_path.write_text(
            json.dumps(
                {
                    "CALIBRATION_PROTOCOL_RECOMMENDATION": "FREEZE",
                    "protocol": {"version": "v2.3-calibration-final", "sha256": sha256_file(self.protocol_path)},
                    "protocol_freeze_gate": {"status": "eligible_for_human_approval"},
                    "final_calibration_status_accounting": {
                        "calibration_images": CALIBRATION_SIZE,
                        "candidate_components": CALIBRATION_SIZE,
                        "uncertain_exclusions": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        self.repeat_path = self.root / "repeat.json"
        self.v21_path = self.root / "v21.json"
        self.v22_path = self.root / "v22.json"
        self.final_path = self.root / "final.json"
        self.analysis_path = self.root / "analysis.json"
        for path in (self.repeat_path, self.v21_path, self.v22_path, self.final_path, self.analysis_path):
            path.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_manifests(self) -> None:
        fields = ["pilot_rank", "stable_source_image_id", "split", "sequence_id", "image_path", "mask_path", "annotation_status", "selection_strata"]
        with self.pilot_manifest.open("w", encoding="utf-8", newline="") as pilot_handle, self.calibration_manifest.open("w", encoding="utf-8", newline="") as calibration_handle, self.component_manifest.open("w", encoding="utf-8", newline="") as component_handle:
            pilot_writer = csv.DictWriter(pilot_handle, fieldnames=fields)
            calibration_writer = csv.DictWriter(calibration_handle, fieldnames=["stable_source_image_id"])
            component_writer = csv.DictWriter(component_handle, fieldnames=["stable_source_image_id", "component_id", "bbox_left", "bbox_top", "bbox_width", "bbox_height"])
            pilot_writer.writeheader()
            calibration_writer.writeheader()
            component_writer.writeheader()
            for index in range(150):
                image_id = f"source-{index:03d}"
                image_relative = f"images/{image_id}.png"
                mask_relative = f"masks/{image_id}.png"
                image_path = self.dataset_root / image_relative
                mask_path = self.dataset_root / mask_relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4)).save(image_path)
                Image.new("L", (4, 4)).save(mask_path)
                pilot_writer.writerow({"pilot_rank": index + 1, "stable_source_image_id": image_id, "split": "train", "sequence_id": f"sequence-{index:03d}", "image_path": image_relative, "mask_path": mask_relative, "annotation_status": "candidate_unreviewed", "selection_strata": "isolated_candidate"})
                component_writer.writerow({"stable_source_image_id": image_id, "component_id": 1, "bbox_left": 0, "bbox_top": 0, "bbox_width": 2, "bbox_height": 2})
                if index < CALIBRATION_SIZE:
                    calibration_writer.writerow({"stable_source_image_id": image_id})

    def test_freeze_and_prepare_exact_remaining_scope(self) -> None:
        freeze_dir = self.root / "freeze"
        freeze = freeze_v23_protocol(
            protocol_path=self.protocol_path,
            calibration_closure_path=self.closure_path,
            boundary_ledger_path=self.ledger_path,
            repeat_state_path=self.repeat_path,
            v21_state_path=self.v21_path,
            v22_state_path=self.v22_path,
            final_state_path=self.final_path,
            final_analysis_path=self.analysis_path,
            output_dir=freeze_dir,
            repository_root=Path(__file__).resolve().parents[1],
        )
        freeze_payload = json.loads(freeze["freeze"].read_text(encoding="utf-8"))
        self.assertEqual(freeze_payload["CALIBRATION_PROTOCOL_STATUS"], PROTOCOL_FREEZE_STATUS)
        self.assertEqual(freeze_payload["frozen_protocol"]["sha256"], sha256_file(freeze["protocol"]))
        with self.assertRaises(FileExistsError):
            freeze_v23_protocol(
                protocol_path=self.protocol_path,
                calibration_closure_path=self.closure_path,
                boundary_ledger_path=self.ledger_path,
                repeat_state_path=self.repeat_path,
                v21_state_path=self.v21_path,
                v22_state_path=self.v22_path,
                final_state_path=self.final_path,
                final_analysis_path=self.analysis_path,
                output_dir=freeze_dir,
                repository_root=Path(__file__).resolve().parents[1],
            )
        prepared = prepare_production_review(
            source_pilot_manifest=self.pilot_manifest,
            component_manifest=self.component_manifest,
            calibration_manifest=self.calibration_manifest,
            frozen_protocol_path=freeze["protocol"],
            protocol_freeze_path=freeze["freeze"],
            calibration_state_path=self.calibration_state_path,
            boundary_ledger_path=self.ledger_path,
            dataset_root=self.dataset_root,
            output_dir=self.root / "production",
        )
        state = load_review_state(prepared["state"])
        scope_ids = state["review_scope"]["image_ids"]
        self.assertEqual(state["protocol"]["version"], "v2.3-calibration-final")
        self.assertEqual(len(scope_ids), PRODUCTION_SIZE)
        self.assertEqual(scope_ids[0], "source-024")
        self.assertNotIn("source-000", scope_ids)
        self.assertEqual(state["images"]["source-024"]["sequence_id"], "sequence-024")
        self.assertEqual(summarize_production_review(state)["production_images_reviewed"], 0)
        self.assertEqual(summarize_production_review(state)["candidate_components_total"], PRODUCTION_SIZE)

        image_id = scope_ids[0]
        image = state["images"][image_id]
        record_annotation(
            state,
            {
                "instance_id": f"{image_id}:rock-001",
                "image_id": image_id,
                "sequence_id": image["sequence_id"],
                "source_candidate_component_id": 1,
                "bbox": [0, 0, 2, 2],
                "polygon": None,
                "annotation_status": BOUNDARY_INDETERMINATE_STATUS,
                "discrete_rock": True,
                "truncated": False,
                "occluded": False,
                "uncertain": False,
                "reviewer_notes": "Visible rock identity is accepted but RGB boundary is not reproducible.",
                "review_version": REVIEW_VERSION,
            },
            reviewer="researcher",
            image_review_status="in_progress",
        )
        finish_image_review(state, image_id, reviewer="researcher")
        report = summarize_production_review(state)
        self.assertEqual(report["boundary_indeterminate"], 1)
        self.assertEqual(report["target_excluded_reasons"], {BOUNDARY_INDETERMINATE_STATUS: 1})
        with self.assertRaisesRegex(ValueError, "boundary-indeterminate accepted objects"):
            maskrcnn_target_for_image(state, image_id, numeric_image_id=1)


if __name__ == "__main__":
    unittest.main()