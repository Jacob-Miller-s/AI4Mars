import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.rock_instance.intra_rater_consistency import (
    component_dispositions,
    mask_iou,
    match_accepted_objects,
    maximum_weight_assignment,
    verify_comparison_provenance,
)


class IntraRaterConsistencyTests(unittest.TestCase):
    def test_mask_iou_uses_intersection_over_union(self) -> None:
        primary = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
        repeat = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
        self.assertEqual(mask_iou(primary, repeat), 1 / 3)

    def test_assignment_is_deterministic_and_one_to_one(self) -> None:
        weights = [[0.8, 0.7], [0.9, 0.1]]
        self.assertEqual(maximum_weight_assignment(weights), [(0, 1), (1, 0)])
        self.assertEqual(maximum_weight_assignment(weights), [(0, 1), (1, 0)])

    def test_component_dispositions_exclude_resolution_and_multi_annotation_structure(self) -> None:
        annotation = lambda instance_id, component_ids: {
            "instance_id": instance_id,
            "annotation_status": "accepted",
            "source_candidate_component_ids": component_ids,
        }
        state = {
            "images": {"image-a": {"candidate_component_ids": [1, 2, 3], "annotations": [annotation("a", [1]), annotation("b", [2]), annotation("c", [2])]}},
            "resolution_records": [{"image_id": "image-a", "source_candidate_component_ids": [3]}],
        }
        self.assertEqual(component_dispositions(state, "image-a"), {
            1: {"kind": "direct", "disposition": "accepted"},
            2: {"kind": "structured", "reason": "multi_annotation_or_multi_component"},
            3: {"kind": "structured", "reason": "explicit_resolution"},
        })

    def test_provenance_rejects_selection_that_does_not_match_primary_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            primary_path = root / "primary.json"
            repeat_path = root / "repeat.json"
            selection_path = root / "selection.json"
            state = {"schema_version": "bad"}
            primary_path.write_text(json.dumps(state), encoding="utf-8")
            repeat_path.write_text(json.dumps(copy.deepcopy(state)), encoding="utf-8")
            selection_path.write_text(json.dumps({"source_primary_state_sha256": "wrong"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_comparison_provenance(primary_path, repeat_path, selection_path)

    def test_matching_reports_unmatched_objects_without_mutating_annotations(self) -> None:
        def accepted(instance_id: str, polygon: list[list[int]]) -> dict:
            return {
                "instance_id": instance_id,
                "annotation_status": "accepted",
                "polygon": polygon,
                "source_candidate_component_ids": [1],
            }

        primary = {"image_width": 6, "image_height": 6, "annotations": [accepted("primary-overlap", [[0, 0], [2, 0], [0, 2]]), accepted("primary-unmatched", [[4, 4], [5, 4], [4, 5]])]}
        repeat = {"image_width": 6, "image_height": 6, "annotations": [accepted("repeat-overlap", [[0, 0], [2, 0], [0, 2]])]}
        primary_before = copy.deepcopy(primary)
        repeat_before = copy.deepcopy(repeat)

        matches, unmatched_primary, unmatched_repeat = match_accepted_objects(primary, repeat)

        self.assertEqual([(match["primary_instance_id"], match["repeat_instance_id"]) for match in matches], [("primary-overlap", "repeat-overlap")])
        self.assertEqual([item["annotation"]["instance_id"] for item in unmatched_primary], ["primary-unmatched"])
        self.assertEqual(unmatched_repeat, [])
        self.assertEqual(primary, primary_before)
        self.assertEqual(repeat, repeat_before)


if __name__ == "__main__":
    unittest.main()