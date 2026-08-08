import unittest

from src.rock_instance.boundary_consistency import _accepted_for_component


class BoundaryConsistencyTests(unittest.TestCase):
    def test_component_lookup_accepts_scalar_and_plural_provenance(self) -> None:
        state = {"images": {"image-a": {"annotations": [
            {"instance_id": "scalar", "annotation_status": "accepted", "source_candidate_component_id": 3},
            {"instance_id": "plural", "annotation_status": "accepted", "source_candidate_component_ids": [3, 4]},
            {"instance_id": "rejected", "annotation_status": "rejected_noise", "source_candidate_component_id": 3},
        ]}}}
        self.assertEqual([item["instance_id"] for item in _accepted_for_component(state, "image-a", 3)], ["scalar", "plural"])


if __name__ == "__main__":
    unittest.main()