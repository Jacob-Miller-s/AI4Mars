import unittest

import numpy as np

from src.rock_instance.component_audit import component_records_for_mask, review_candidates


class ComponentAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "split": "train",
            "stable_source_image_id": "source-a",
            "sequence_id": "sequence-a",
            "dataset_relative_image_path": "msl/ncam/images/edr/source-a.JPG",
            "dataset_relative_mask_path": "msl/ncam/labels/train/source-a.png",
        }

    def test_eight_connectivity_merges_diagonal_candidate_pixels(self) -> None:
        mask = np.array([[3, 0, 0], [0, 3, 1], [0, 0, 0]], dtype=np.uint8)

        components, image = component_records_for_mask(mask, self.row, connectivity=8, tiny_area_pixels=1)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["area_pixels"], 2)
        self.assertEqual(image["bedrock_adjacent_pixels"], 1)
        self.assertIn("bedrock_big_rock_boundary", image["manual_review_reasons"])

    def test_four_connectivity_keeps_diagonal_candidate_pixels_separate(self) -> None:
        mask = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 0]], dtype=np.uint8)

        components, image = component_records_for_mask(mask, self.row, connectivity=4, tiny_area_pixels=1)

        self.assertEqual(len(components), 2)
        self.assertTrue(image["multiple_components"])

    def test_review_queue_is_deterministic_and_candidate_only(self) -> None:
        no_signal = {"manual_review_reasons": "", "manual_review_priority": 0, "split": "train", "sequence_id": "A", "stable_source_image_id": "A"}
        lower_priority = {"manual_review_reasons": "tiny_component", "manual_review_priority": 1, "split": "val", "sequence_id": "B", "stable_source_image_id": "B"}
        higher_priority = {"manual_review_reasons": "multiple_semantic_components", "manual_review_priority": 4, "split": "train", "sequence_id": "C", "stable_source_image_id": "C"}

        queue = review_candidates([lower_priority, no_signal, higher_priority])

        self.assertEqual([row["stable_source_image_id"] for row in queue], ["C", "B"])


if __name__ == "__main__":
    unittest.main()