import copy
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from PIL import Image

from src.rock_instance.annotations import (
    BOUNDARY_INDETERMINATE_STATUS,
    PRODUCTION_COMPONENT_COUNT,
    PRODUCTION_SIZE,
    REVIEW_VERSION,
    finish_image_review,
    load_review_state,
    maskrcnn_target_for_image,
    ordinary_maskrcnn_target_eligibility,
    record_annotation,
    record_resolution,
    replace_annotation,
    save_review_state,
    sha256_file,
    validate_production_review_provenance,
)
from src.rock_instance.production_review import freeze_v23_protocol, summarize_production_review
from src.rock_instance.review_tool import (
    RockInstanceReviewUI,
    next_unreviewed_image_id,
    validated_candidate_components,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "rock_instance"
PRODUCTION_ROOT = ARTIFACT_ROOT / "production_review_v2.3"
STATE_PATH = PRODUCTION_ROOT / "review_state.json"
COMPONENT_MANIFEST = PRODUCTION_ROOT / "big_rock_component_candidates.csv"
BOUNDARY_LEDGER = ARTIFACT_ROOT / "calibration_finalization_v2.3" / "boundary_indeterminate_exclusions.json"


class ProductionReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = load_review_state(STATE_PATH)

    def _image_id(self, *, minimum_components: int = 1, excluded: set[str] | None = None) -> str:
        excluded = excluded or set()
        return next(
            image_id
            for image_id in self.state["review_scope"]["image_ids"]
            if image_id not in excluded
            and len(self.state["images"][image_id]["candidate_component_ids"]) >= minimum_components
        )

    def _annotation(
        self,
        image_id: str,
        instance_id: str,
        component_ids: list[int],
        *,
        status: str = "accepted",
        polygon: list[list[float]] | None = None,
    ) -> dict:
        if polygon is None and status == "accepted":
            polygon = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
        return {
            "instance_id": instance_id,
            "image_id": image_id,
            "sequence_id": self.state["images"][image_id]["sequence_id"],
            "source_candidate_component_id": component_ids[0],
            "source_candidate_component_ids": component_ids,
            "bbox": [0, 0, 3, 3],
            "polygon": polygon,
            "annotation_status": status,
            "discrete_rock": status in {"accepted", BOUNDARY_INDETERMINATE_STATUS},
            "truncated": False,
            "occluded": False,
            "uncertain": status == "uncertain",
            "reviewer_notes": "review evidence",
            "review_version": REVIEW_VERSION,
        }

    def test_frozen_population_opens_at_zero_of_126_with_814_candidates(self) -> None:
        progress = summarize_production_review(self.state)

        self.assertEqual(progress["production_images_total"], PRODUCTION_SIZE)
        self.assertEqual(progress["production_images_reviewed"], 0)
        self.assertEqual(progress["production_images_remaining"], PRODUCTION_SIZE)
        self.assertEqual(progress["candidate_components_total"], PRODUCTION_COMPONENT_COUNT)
        self.assertEqual(progress["candidate_components_unresolved"], PRODUCTION_COMPONENT_COUNT)
        self.assertEqual(next_unreviewed_image_id(self.state), self.state["review_scope"]["image_ids"][0])
        self.assertTrue(
            all(
                self.state["images"][image_id]["review_status"] == "unreviewed"
                and not self.state["images"][image_id]["annotations"]
                for image_id in self.state["review_scope"]["image_ids"]
            )
        )

    def test_provenance_is_valid_and_all_required_mismatches_fail_closed(self) -> None:
        validate_production_review_provenance(self.state)
        image_id = self._image_id()
        component_id = self.state["images"][image_id]["candidate_component_ids"][0]
        annotation = self._annotation(image_id, f"{image_id}:rock-001", [component_id])

        def protocol_mismatch(state: dict) -> None:
            state["protocol"]["sha256"] = "0" * 64

        def calibration_mismatch(state: dict) -> None:
            state["production_review"]["calibration_scope_sha256"] = "0" * 64

        def source_pilot_mismatch(state: dict) -> None:
            state["pilot_id"] = "wrong-pilot"

        def exclusion_ledger_mismatch(state: dict) -> None:
            state["production_review"]["boundary_ledger_sha256"] = "0" * 64

        def missing_provenance(state: dict) -> None:
            del state["production_review"]["protocol_freeze_sha256"]

        def missing_scope(state: dict) -> None:
            del state["review_scope"]
            del state["component_review"]

        def missing_discriminator(state: dict) -> None:
            del state["production_review_schema_version"]

        def image_path_mismatch(state: dict) -> None:
            state["images"][image_id]["image_path"] = "different/source.JPG"

        for name, mutate in (
            ("protocol", protocol_mismatch),
            ("calibration", calibration_mismatch),
            ("source pilot", source_pilot_mismatch),
            ("exclusion ledger", exclusion_ledger_mismatch),
            ("missing", missing_provenance),
            ("missing scope", missing_scope),
            ("missing discriminator", missing_discriminator),
            ("image path", image_path_mismatch),
        ):
            with self.subTest(name=name):
                stale_state = copy.deepcopy(self.state)
                mutate(stale_state)
                with self.assertRaisesRegex(ValueError, "Frozen production review"):
                    record_annotation(stale_state, annotation, reviewer="researcher")
                self.assertEqual(stale_state["images"][image_id]["annotations"], [])

    def test_direct_save_cannot_bypass_provenance_validation(self) -> None:
        image_id = self._image_id()
        component_id = self.state["images"][image_id]["candidate_component_ids"][0]
        self.state["images"][image_id]["annotations"].append(
            self._annotation(image_id, f"{image_id}:rock-001", [component_id])
        )
        self.state["production_review"]["source_pilot_manifest_sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "Frozen production review"):
                save_review_state(Path(temporary_directory) / "review_state.json", self.state)

    def test_invalid_annotation_status_argument_does_not_mutate_state(self) -> None:
        image_id = self._image_id()
        component_id = self.state["images"][image_id]["candidate_component_ids"][0]
        annotation = self._annotation(image_id, f"{image_id}:rock-001", [component_id])

        with self.assertRaisesRegex(ValueError, "image_review_status"):
            record_annotation(
                self.state,
                annotation,
                reviewer="researcher",
                image_review_status="invalid",
            )

        image = self.state["images"][image_id]
        self.assertEqual(image["annotations"], [])
        self.assertEqual(image["review_status"], "unreviewed")
        self.assertIsNone(image["reviewer"])

    def test_save_resume_and_polygon_edit_preserve_geometry(self) -> None:
        image_id = self._image_id()
        component_id = self.state["images"][image_id]["candidate_component_ids"][0]
        instance_id = f"{image_id}:rock-001"
        record_annotation(
            self.state,
            self._annotation(image_id, instance_id, [component_id]),
            reviewer="researcher",
            image_review_status="in_progress",
        )
        edited = self._annotation(
            image_id,
            instance_id,
            [component_id],
            polygon=[[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]],
        )
        edited["bbox"] = [1, 1, 4, 4]
        replace_annotation(self.state, edited, reviewer="researcher")

        with tempfile.TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "review_state.json"
            save_review_state(state_path, self.state)
            resumed = load_review_state(state_path)

        saved = resumed["images"][image_id]["annotations"][0]
        self.assertEqual(saved["polygon"], edited["polygon"])
        self.assertEqual(saved["bbox"], edited["bbox"])
        self.assertEqual(saved["source_candidate_component_ids"], [component_id])

    def test_invalid_polygon_and_out_of_scope_components_are_rejected(self) -> None:
        image_id = self._image_id()
        component_id = self.state["images"][image_id]["candidate_component_ids"][0]
        malformed = self._annotation(
            image_id,
            f"{image_id}:rock-001",
            [component_id],
            polygon=[[0.0, 0.0], [1.0, 1.0]],
        )
        with self.assertRaisesRegex(ValueError, "at least three"):
            record_annotation(self.state, malformed, reviewer="researcher")

        outside_scope = next(image_id for image_id in self.state["images"] if image_id not in self.state["review_scope"]["image_ids"])
        outside = self._annotation(
            image_id,
            f"{outside_scope}:rock-001",
            [component_id],
        )
        outside["image_id"] = outside_scope
        outside["sequence_id"] = self.state["images"][outside_scope]["sequence_id"]
        with self.assertRaisesRegex(ValueError, "approved remaining-image scope"):
            record_annotation(self.state, outside, reviewer="researcher")

        unknown_component = self._annotation(
            image_id,
            f"{image_id}:rock-002",
            [max(self.state["images"][image_id]["candidate_component_ids"]) + 1],
        )
        with self.assertRaisesRegex(ValueError, "approved candidate components"):
            record_annotation(self.state, unknown_component, reviewer="researcher")

    def test_merge_and_split_lineage_survive_resume(self) -> None:
        merge_image_id = self._image_id(minimum_components=3)
        merge_components = self.state["images"][merge_image_id]["candidate_component_ids"][:2]
        merge_instance_id = f"{merge_image_id}:rock-001"
        record_annotation(
            self.state,
            self._annotation(merge_image_id, merge_instance_id, merge_components),
            reviewer="researcher",
            image_review_status="in_progress",
        )
        unrelated_component = self.state["images"][merge_image_id]["candidate_component_ids"][2]
        invalid_merge = {
            "resolution_id": f"{merge_image_id}:merge-invalid",
            "resolution_type": "merge",
            "image_id": merge_image_id,
            "sequence_id": self.state["images"][merge_image_id]["sequence_id"],
            "source_candidate_component_ids": merge_components,
            "initial_decision_instance_ids": [
                f"{merge_image_id}:component-{unrelated_component}"
            ],
            "resolved_annotation_instance_ids": [merge_instance_id],
            "reviewer_notes": "Invalid unrelated source.",
        }
        with self.assertRaisesRegex(ValueError, "exactly match"):
            record_resolution(self.state, invalid_merge)
        record_resolution(
            self.state,
            {
                "resolution_id": f"{merge_image_id}:merge-001",
                "resolution_type": "merge",
                "image_id": merge_image_id,
                "sequence_id": self.state["images"][merge_image_id]["sequence_id"],
                "source_candidate_component_ids": merge_components,
                "initial_decision_instance_ids": [
                    f"{merge_image_id}:component-{component_id}"
                    for component_id in merge_components
                ],
                "resolved_annotation_instance_ids": [merge_instance_id],
                "reviewer_notes": "Two semantic components are one physical rock.",
            },
        )

        split_image_id = self._image_id(excluded={merge_image_id})
        split_component = self.state["images"][split_image_id]["candidate_component_ids"][0]
        split_instance_ids = [f"{split_image_id}:rock-001", f"{split_image_id}:rock-002"]
        for index, instance_id in enumerate(split_instance_ids):
            annotation = self._annotation(split_image_id, instance_id, [split_component])
            annotation["polygon"] = [
                [float(index * 3), 0.0],
                [float(index * 3 + 2), 0.0],
                [float(index * 3), 2.0],
            ]
            annotation["bbox"] = [index * 3, 0, 3, 3]
            record_annotation(
                self.state,
                annotation,
                reviewer="researcher",
                image_review_status="in_progress",
            )
        record_resolution(
            self.state,
            {
                "resolution_id": f"{split_image_id}:split-001",
                "resolution_type": "split",
                "image_id": split_image_id,
                "sequence_id": self.state["images"][split_image_id]["sequence_id"],
                "source_candidate_component_ids": [split_component],
                "initial_decision_instance_ids": [f"{split_image_id}:component-{split_component}"],
                "resolved_annotation_instance_ids": split_instance_ids,
                "reviewer_notes": "One semantic component contains two visible physical rocks.",
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "review_state.json"
            save_review_state(state_path, self.state)
            resumed = load_review_state(state_path)

        records = {record["resolution_type"]: record for record in resumed["resolution_records"]}
        self.assertEqual(records["merge"]["source_candidate_component_ids"], merge_components)
        self.assertEqual(records["merge"]["resolved_annotation_instance_ids"], [merge_instance_id])
        self.assertEqual(records["split"]["source_candidate_component_ids"], [split_component])
        self.assertEqual(records["split"]["resolved_annotation_instance_ids"], split_instance_ids)

    def test_nonaccepted_resolution_cannot_cover_unlinked_component(self) -> None:
        image_id = self._image_id(minimum_components=2)
        component_ids = self.state["images"][image_id]["candidate_component_ids"][:2]
        instance_id = f"{image_id}:rock-001"
        record_annotation(
            self.state,
            self._annotation(image_id, instance_id, [component_ids[0]], status="uncertain"),
            reviewer="researcher",
            image_review_status="in_progress",
        )

        with self.assertRaisesRegex(ValueError, "preserve every source component"):
            record_resolution(
                self.state,
                {
                    "resolution_id": f"{image_id}:merge-invalid",
                    "resolution_type": "merge",
                    "image_id": image_id,
                    "sequence_id": self.state["images"][image_id]["sequence_id"],
                    "source_candidate_component_ids": component_ids,
                    "initial_decision_instance_ids": [
                        f"{image_id}:component-{component_id}"
                        for component_id in component_ids
                    ],
                    "resolved_annotation_instance_ids": [instance_id],
                    "reviewer_notes": "Invalid incomplete component linkage.",
                },
            )

    def test_conflicting_direct_dispositions_cannot_finish_image(self) -> None:
        image_id = next(
            image_id
            for image_id in self.state["review_scope"]["image_ids"]
            if len(self.state["images"][image_id]["candidate_component_ids"]) == 1
        )
        component_id = self.state["images"][image_id]["candidate_component_ids"][0]
        for index, status in enumerate(("rejected_noise", "uncertain"), start=1):
            record_annotation(
                self.state,
                self._annotation(
                    image_id,
                    f"{image_id}:rock-{index:03d}",
                    [component_id],
                    status=status,
                ),
                reviewer="researcher",
                image_review_status="in_progress",
            )

        with self.assertRaisesRegex(ValueError, "matching split record"):
            finish_image_review(self.state, image_id, reviewer="researcher")
        self.assertEqual(self.state["images"][image_id]["review_status"], "in_progress")

    def test_unreviewed_production_image_cannot_become_an_empty_target(self) -> None:
        image_id = self._image_id()

        self.assertEqual(
            ordinary_maskrcnn_target_eligibility(self.state, image_id)["reason"],
            "review_incomplete",
        )
        with self.assertRaisesRegex(ValueError, "complete review"):
            maskrcnn_target_for_image(self.state, image_id, numeric_image_id=1)

    def test_uncertainty_and_boundary_indeterminate_remain_distinct_after_resume(self) -> None:
        uncertainty_state = copy.deepcopy(self.state)
        uncertain_image_id = next(
            image_id
            for image_id in uncertainty_state["review_scope"]["image_ids"]
            if len(uncertainty_state["images"][image_id]["candidate_component_ids"]) == 1
        )
        uncertain_component = uncertainty_state["images"][uncertain_image_id]["candidate_component_ids"][0]
        record_annotation(
            uncertainty_state,
            self._annotation(
                uncertain_image_id,
                f"{uncertain_image_id}:rock-001",
                [uncertain_component],
                status="uncertain",
            ),
            reviewer="researcher",
            image_review_status="in_progress",
        )
        finish_image_review(uncertainty_state, uncertain_image_id, reviewer="researcher")

        boundary_state = copy.deepcopy(self.state)
        boundary_image_id = next(
            image_id
            for image_id in boundary_state["review_scope"]["image_ids"]
            if len(boundary_state["images"][image_id]["candidate_component_ids"]) == 1
        )
        boundary_component = boundary_state["images"][boundary_image_id]["candidate_component_ids"][0]
        record_annotation(
            boundary_state,
            self._annotation(
                boundary_image_id,
                f"{boundary_image_id}:rock-001",
                [boundary_component],
                status=BOUNDARY_INDETERMINATE_STATUS,
            ),
            reviewer="researcher",
            image_review_status="in_progress",
        )
        finish_image_review(boundary_state, boundary_image_id, reviewer="researcher")

        with tempfile.TemporaryDirectory() as temporary_directory:
            uncertain_path = Path(temporary_directory) / "uncertain.json"
            boundary_path = Path(temporary_directory) / "boundary.json"
            save_review_state(uncertain_path, uncertainty_state)
            save_review_state(boundary_path, boundary_state)
            resumed_uncertain = load_review_state(uncertain_path)
            resumed_boundary = load_review_state(boundary_path)

        self.assertEqual(
            ordinary_maskrcnn_target_eligibility(resumed_uncertain, uncertain_image_id)["reason"],
            "uncertain",
        )
        self.assertEqual(
            ordinary_maskrcnn_target_eligibility(resumed_boundary, boundary_image_id)["reason"],
            BOUNDARY_INDETERMINATE_STATUS,
        )
        self.assertEqual(summarize_production_review(resumed_boundary)["production_images_reviewed"], 1)
        with self.assertRaisesRegex(ValueError, "boundary-indeterminate accepted objects"):
            maskrcnn_target_for_image(resumed_boundary, boundary_image_id, numeric_image_id=1)

    def test_approved_exclusion_ledger_stays_outside_editable_scope(self) -> None:
        ledger = json.loads(BOUNDARY_LEDGER.read_text(encoding="utf-8"))
        excluded_ids = {record["image_id"] for record in ledger["boundary_indeterminate_records"]}

        self.assertTrue(excluded_ids)
        self.assertTrue(excluded_ids.isdisjoint(self.state["review_scope"]["image_ids"]))

    def test_approved_calibration_evidence_remains_loadable_as_calibration(self) -> None:
        calibration_state = load_review_state(
            ARTIFACT_ROOT / "calibration_resolved_v2" / "review_state.json"
        )

        self.assertEqual(calibration_state["review_scope"]["name"], "calibration")

    def test_reviewer_displays_frozen_population_and_uses_approved_components(self) -> None:
        image_id = self._image_id()
        image = self.state["images"][image_id]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            state_path = temporary_root / "review_state.json"
            dataset_root = temporary_root / "dataset"
            image_path = dataset_root / image["image_path"]
            mask_path = dataset_root / image["mask_path"]
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            Image.new("RGB", (image["image_width"], image["image_height"])).save(image_path)
            Image.new("L", (image["image_width"], image["image_height"])).save(mask_path)
            save_review_state(state_path, self.state)

            ui = RockInstanceReviewUI(
                state_path=state_path,
                component_candidates_csv=COMPONENT_MANIFEST,
                dataset_root=dataset_root,
                image_id=image_id,
                reviewer="researcher",
            )
            try:
                title = ui.figure._suptitle.get_text()
                self.assertIn("0/126 reviewed", title)
                self.assertIn("814 candidate components", title)
                self.assertEqual(
                    [int(component["component_id"]) for component in ui.components],
                    self.state["images"][image_id]["candidate_component_ids"],
                )
            finally:
                matplotlib.pyplot.close(ui.figure)

        with tempfile.TemporaryDirectory() as temporary_directory:
            bad_manifest = Path(temporary_directory) / "components.csv"
            bad_manifest.write_text(
                "stable_source_image_id,component_id\n"
                f"{image_id},{self.state['images'][image_id]['candidate_component_ids'][0]}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "approved provenance"):
                validated_candidate_components(self.state, bad_manifest, image_id)

    def test_protocol_freeze_serialization_is_platform_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            protocol = root / "annotation_protocol_v2.3-calibration-final.md"
            protocol.write_text("# v2.3-calibration-final\n", encoding="utf-8")
            ledger = root / "ledger.json"
            ledger.write_text(
                json.dumps({"schema_version": "rock_instance_calibration_finalization_v1"}),
                encoding="utf-8",
            )
            closure = root / "closure.json"
            closure.write_text(
                json.dumps(
                    {
                        "CALIBRATION_PROTOCOL_RECOMMENDATION": "FREEZE",
                        "protocol": {
                            "version": "v2.3-calibration-final",
                            "sha256": sha256_file(protocol),
                        },
                        "protocol_freeze_gate": {"status": "eligible_for_human_approval"},
                        "final_calibration_status_accounting": {
                            "calibration_images": 24,
                            "candidate_components": 173,
                            "uncertain_exclusions": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            evidence_paths = [root / f"evidence-{index}.json" for index in range(5)]
            for path in evidence_paths:
                path.write_text("{}\n", encoding="utf-8")

            frozen = freeze_v23_protocol(
                protocol_path=protocol,
                calibration_closure_path=closure,
                boundary_ledger_path=ledger,
                repeat_state_path=evidence_paths[0],
                v21_state_path=evidence_paths[1],
                v22_state_path=evidence_paths[2],
                final_state_path=evidence_paths[3],
                final_analysis_path=evidence_paths[4],
                output_dir=root / "freeze",
                repository_root=REPOSITORY_ROOT,
            )

            for path in (frozen["freeze"], frozen["closure_report"]):
                data = path.read_bytes()
                self.assertIn(b"\r\n", data)
                self.assertNotIn(b"\n", data.replace(b"\r\n", b""))


if __name__ == "__main__":
    unittest.main()
