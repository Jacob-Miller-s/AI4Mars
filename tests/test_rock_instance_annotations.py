import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.rock_instance.annotations import (
    REVIEW_VERSION,
    component_coverage_for_image,
    configure_component_review,
    configure_review_scope,
    finish_image_review,
    initial_calibration_reference,
    initialize_review_state,
    load_review_state,
    maskrcnn_target_for_image,
    record_annotation,
    record_resolution,
    save_review_state,
    set_candidate_independent_observation,
    sha256_file,
    unresolved_candidate_component_ids,
)
from src.rock_instance.review_tool import (
    _bbox_from_polygon,
    initialize_calibration_resolution_artifacts,
    next_unreviewed_image_id,
    restart_in_progress_image,
)
from src.rock_instance.review_report import summarize_review_state
from src.rock_instance.repeat_review import initialize_isolated_repeat_state, select_repeat_image_ids


class AnnotationFixture(unittest.TestCase):
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
        self.manifest = self.root / "candidates.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pilot_rank", "stable_source_image_id", "split", "sequence_id", "image_path", "mask_path", "annotation_status", "selection_strata"])
            writer.writeheader()
            writer.writerow({"pilot_rank": 1, "stable_source_image_id": "source-a", "split": "train", "sequence_id": "sequence-a", "image_path": self.image_relative, "mask_path": self.mask_relative, "annotation_status": "candidate_unreviewed", "selection_strata": "isolated_candidate"})
        self.state = initialize_review_state(self.manifest, self.dataset_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _annotation(self, instance_id: str = "source-a:1") -> dict:
        return {"instance_id": instance_id, "image_id": "source-a", "sequence_id": "sequence-a", "source_candidate_component_id": 1, "bbox": [1, 2, 5, 4], "polygon": [[1, 2], [6, 2], [6, 6], [1, 6]], "annotation_status": "accepted", "discrete_rock": True, "truncated": False, "occluded": False, "uncertain": False, "reviewer_notes": "visually discrete", "review_version": REVIEW_VERSION}

    def _terminal_annotation(self, instance_id: str, component_ids: list[int], *, status: str) -> dict:
        annotation = self._annotation(instance_id)
        annotation["source_candidate_component_id"] = component_ids[0]
        annotation["source_candidate_component_ids"] = component_ids
        annotation["annotation_status"] = status
        annotation["discrete_rock"] = status == "accepted"
        annotation["uncertain"] = status == "uncertain"
        if status != "accepted":
            annotation.pop("polygon")
        return annotation

    def _corrected_calibration_state(self, component_ids: list[int]) -> tuple[dict, Path]:
        calibration_manifest = self.root / "calibration.csv"
        with calibration_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stable_source_image_id"])
            writer.writeheader()
            writer.writerow({"stable_source_image_id": "source-a"})
        component_manifest = self.root / "components.csv"
        component_manifest.write_text("stable_source_image_id,component_id\n" + "\n".join(f"source-a,{component_id}" for component_id in component_ids) + "\n", encoding="utf-8")
        protocol_path = self.root / "annotation_protocol_v2.md"
        protocol_path.write_text("# v2.0-calibration-resolved\n", encoding="utf-8")
        initial_state = initialize_review_state(self.manifest, self.dataset_root)
        initial_annotation = self._terminal_annotation("initial:source-a", [component_ids[0]], status="rejected_noise")
        record_annotation(initial_state, initial_annotation, reviewer="initial_reviewer")
        snapshot_path = self.root / "review_state_initial_v1.json"
        save_review_state(snapshot_path, initial_state)
        state = initialize_review_state(self.manifest, self.dataset_root, pilot_id="calibration_resolved_v2")
        configure_review_scope(state, name="calibration", image_ids=["source-a"], source_manifest=calibration_manifest)
        configure_component_review(
            state,
            component_manifest=component_manifest,
            component_ids_by_image={"source-a": component_ids},
            protocol_path=protocol_path,
            initial_calibration_reference=initial_calibration_reference(snapshot_path),
        )
        return state, snapshot_path

    def test_state_save_resume_and_maskrcnn_target_shapes(self) -> None:
        record_annotation(self.state, self._annotation(), reviewer="researcher")
        state_path = self.root / "review_state.json"
        save_review_state(state_path, self.state)
        restored = load_review_state(state_path)
        target = maskrcnn_target_for_image(restored, "source-a", numeric_image_id=7)

        self.assertEqual(target["boxes"].dtype, torch.float32)
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(target["masks"].dtype, torch.bool)
        self.assertEqual(tuple(target["boxes"].shape), (1, 4))
        self.assertEqual(tuple(target["masks"].shape), (1, 10, 12))
        self.assertEqual(target["image_id"].item(), 7)

    def test_duplicate_instance_id_and_malformed_bbox_are_rejected(self) -> None:
        record_annotation(self.state, self._annotation(), reviewer="researcher")
        with self.assertRaisesRegex(ValueError, "Duplicate instance_id"):
            record_annotation(self.state, self._annotation(), reviewer="researcher")
        malformed = self._annotation("source-a:2")
        malformed["bbox"] = [10, 2, 5, 4]
        with self.assertRaisesRegex(ValueError, "beyond image"):
            record_annotation(self.state, malformed, reviewer="researcher")

    def test_mask_image_geometry_mismatch_is_rejected(self) -> None:
        Image.new("L", (11, 10), color=0).save(self.dataset_root / self.mask_relative)
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            initialize_review_state(self.manifest, self.dataset_root)

    def test_report_does_not_estimate_burden_before_manual_review(self) -> None:
        report = summarize_review_state(self.state, calibration_image_ids={"source-a"})
        self.assertEqual(report["images_reviewed"], 0)
        self.assertIsNone(report["calibration"]["annotation_burden_estimate"])

    def test_calibration_scope_limits_the_default_review_queue(self) -> None:
        second_image = dict(self.state["images"]["source-a"])
        second_image.update({"image_id": "source-b", "pilot_rank": 2, "sequence_id": "sequence-b", "annotations": []})
        self.state["images"]["source-b"] = second_image
        calibration_manifest = self.root / "calibration.csv"
        with calibration_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stable_source_image_id"])
            writer.writeheader()
            writer.writerow({"stable_source_image_id": "source-b"})

        configure_review_scope(
            self.state,
            name="calibration",
            image_ids=["source-b"],
            source_manifest=calibration_manifest,
        )

        self.assertEqual(next_unreviewed_image_id(self.state), "source-b")

    def test_interactive_decisions_remain_in_progress_until_explicit_finish(self) -> None:
        record_annotation(
            self.state,
            self._annotation(),
            reviewer="researcher",
            image_review_status="in_progress",
        )

        self.assertEqual(self.state["images"]["source-a"]["review_status"], "in_progress")
        self.assertEqual(next_unreviewed_image_id(self.state), "source-a")

        finish_image_review(self.state, "source-a", reviewer="researcher")
        self.assertEqual(self.state["images"]["source-a"]["review_status"], "reviewed")
        with self.assertRaisesRegex(ValueError, "All images"):
            next_unreviewed_image_id(self.state)

    def test_restart_archives_incomplete_image_attempts_before_resetting_it(self) -> None:
        record_annotation(
            self.state,
            self._annotation(),
            reviewer="researcher",
            image_review_status="in_progress",
        )
        state_path = self.root / "review_state.json"
        save_review_state(state_path, self.state)

        archive_path = restart_in_progress_image(state_path, "source-a", reason="Restart after recording an incorrect disposition.")
        restored = load_review_state(state_path)

        self.assertTrue(archive_path.is_file())
        self.assertIn("source-a:1", archive_path.read_text(encoding="utf-8"))
        self.assertEqual(restored["images"]["source-a"]["annotations"], [])
        self.assertEqual(restored["images"]["source-a"]["review_status"], "unreviewed")

    def test_polygon_bbox_includes_its_last_rasterized_pixel(self) -> None:
        self.assertEqual(_bbox_from_polygon([[1.0, 2.0], [6.0, 2.0], [6.0, 6.0], [1.0, 6.0]]), [1, 2, 6, 5])

    def test_corrected_calibration_blocks_completion_until_all_components_have_terminal_coverage(self) -> None:
        state, _ = self._corrected_calibration_state([1, 2])
        record_annotation(state, self._terminal_annotation("resolved:1", [1], status="rejected_noise"), reviewer="researcher")

        self.assertEqual(state["images"]["source-a"]["review_status"], "in_progress")
        self.assertEqual(unresolved_candidate_component_ids(state, "source-a"), [2])
        with self.assertRaisesRegex(ValueError, r"unresolved candidate component IDs: \[2\]"):
            finish_image_review(state, "source-a", reviewer="researcher")

        record_annotation(state, self._terminal_annotation("resolved:2", [2], status="uncertain"), reviewer="researcher", image_review_status="in_progress")
        self.assertEqual(component_coverage_for_image(state, "source-a"), {1, 2})
        finish_image_review(state, "source-a", reviewer="researcher")
        with self.assertRaisesRegex(ValueError, "excluded from ordinary Mask R-CNN targets"):
            maskrcnn_target_for_image(state, "source-a", numeric_image_id=1)

    def test_merge_resolution_preserves_plural_components_and_initial_links(self) -> None:
        state, snapshot_path = self._corrected_calibration_state([1, 2])
        merged = self._terminal_annotation("resolved:merge", [1, 2], status="accepted")
        record_annotation(state, merged, reviewer="researcher", image_review_status="in_progress")
        record_resolution(
            state,
            {
                "resolution_id": "resolution:merge:001",
                "resolution_type": "merge",
                "image_id": "source-a",
                "sequence_id": "sequence-a",
                "source_candidate_component_ids": [1, 2],
                "initial_decision_instance_ids": ["initial:source-a"],
                "resolved_annotation_instance_ids": ["resolved:merge"],
                "reviewer_notes": "Human reviewer confirmed one object across semantic fragments.",
            },
        )
        finish_image_review(state, "source-a", reviewer="researcher")
        save_review_state(self.root / "corrected_state.json", state)

        self.assertEqual(state["images"]["source-a"]["annotations"][0]["source_candidate_component_ids"], [1, 2])
        self.assertEqual(state["resolution_records"][0]["initial_decision_instance_ids"], ["initial:source-a"])
        self.assertEqual(state["initial_calibration_reference"]["snapshot_sha256"], sha256_file(snapshot_path))

    def test_split_resolution_preserves_parent_component_for_each_child(self) -> None:
        state, _ = self._corrected_calibration_state([1])
        first_child = self._terminal_annotation("resolved:split:1", [1], status="accepted")
        second_child = self._terminal_annotation("resolved:split:2", [1], status="accepted")
        record_annotation(state, first_child, reviewer="researcher", image_review_status="in_progress")
        record_annotation(state, second_child, reviewer="researcher", image_review_status="in_progress")
        record_resolution(
            state,
            {
                "resolution_id": "resolution:split:001",
                "resolution_type": "split",
                "image_id": "source-a",
                "sequence_id": "sequence-a",
                "source_candidate_component_ids": [1],
                "initial_decision_instance_ids": ["initial:source-a"],
                "resolved_annotation_instance_ids": ["resolved:split:1", "resolved:split:2"],
                "reviewer_notes": "Human reviewer resolved two visible rocks from one semantic component.",
            },
        )
        finish_image_review(state, "source-a", reviewer="researcher")
        self.assertEqual(unresolved_candidate_component_ids(state, "source-a"), [])

    def test_candidate_independent_observation_does_not_create_target_or_annotation(self) -> None:
        state, _ = self._corrected_calibration_state([1])
        set_candidate_independent_observation(
            state, "source-a", observed=True, note="Obvious rock outside all semantic candidates; no boundary recorded."
        )

        self.assertTrue(state["images"]["source-a"]["obvious_candidate_independent_rock_observed"])
        self.assertEqual(state["images"]["source-a"]["annotations"], [])

    def test_repeat_selection_is_deterministic_and_repeat_state_is_isolated(self) -> None:
        state, _ = self._corrected_calibration_state([1])
        record_annotation(state, self._terminal_annotation("resolved:1", [1], status="rejected_noise"), reviewer="researcher", image_review_status="in_progress")
        finish_image_review(state, "source-a", reviewer="researcher")

        selected = select_repeat_image_ids(state, target_size=1)
        repeat_state = initialize_isolated_repeat_state(state, selected)
        record_annotation(repeat_state, self._terminal_annotation("repeat:1", [1], status="rejected_noise"), reviewer="repeat_reviewer")

        self.assertEqual(selected, select_repeat_image_ids(state, target_size=1, seed=42))
        self.assertEqual(repeat_state["review_scope"]["name"], "calibration_repeat")
        self.assertEqual(state["images"]["source-a"]["annotations"][0]["instance_id"], "resolved:1")
        self.assertEqual(repeat_state["images"]["source-a"]["annotations"][0]["instance_id"], "repeat:1")

    def test_fresh_corrected_state_references_but_never_changes_initial_snapshot(self) -> None:
        initial_state = initialize_review_state(self.manifest, self.dataset_root)
        record_annotation(
            initial_state,
            self._terminal_annotation("initial:source-a", [1], status="rejected_noise"),
            reviewer="initial_reviewer",
        )
        snapshot_path = self.root / "review_state_initial_v1.json"
        save_review_state(snapshot_path, initial_state)
        initial_hash = sha256_file(snapshot_path)
        calibration_manifest = self.root / "calibration.csv"
        calibration_manifest.write_text("stable_source_image_id\nsource-a\n", encoding="utf-8")
        component_manifest = self.root / "components.csv"
        component_manifest.write_text("stable_source_image_id,component_id\nsource-a,1\n", encoding="utf-8")
        protocol_path = self.root / "annotation_protocol_v2.md"
        protocol_path.write_text("# v2.0-calibration-resolved\n", encoding="utf-8")

        state_path = initialize_calibration_resolution_artifacts(
            candidate_manifest=self.manifest,
            component_candidates_csv=component_manifest,
            calibration_manifest=calibration_manifest,
            initial_snapshot=snapshot_path,
            protocol_path=protocol_path,
            dataset_root=self.dataset_root,
            output_dir=self.root / "calibration_resolved_v2",
        )
        corrected_state = load_review_state(state_path)

        self.assertEqual(sha256_file(snapshot_path), initial_hash)
        self.assertEqual(corrected_state["pilot_id"], "calibration_resolved_v2")
        self.assertEqual(corrected_state["initial_calibration_reference"]["snapshot_sha256"], initial_hash)
        self.assertEqual(corrected_state["images"]["source-a"]["annotations"], [])


if __name__ == "__main__":
    unittest.main()