import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.rock_instance.annotations import (
    REVIEW_VERSION,
    initialize_review_state,
    load_review_state,
    maskrcnn_target_for_image,
    record_annotation,
    save_review_state,
)
from src.rock_instance.review_report import summarize_review_state


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


if __name__ == "__main__":
    unittest.main()