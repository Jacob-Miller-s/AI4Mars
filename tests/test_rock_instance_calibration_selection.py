import unittest

from src.rock_instance.calibration_selection import select_calibration_records


class CalibrationSelectionTests(unittest.TestCase):
    def _record(self, index: int) -> dict[str, str]:
        strata = ["rgb_only_or_geometry_unresolved"]
        mapping = {
            0: "isolated_candidate", 1: "multiple_candidate_regions", 2: "bedrock_big_rock_boundary",
            3: "very_large_component_candidate", 4: "tiny_component_candidate", 5: "border_truncation_candidate",
        }
        strata.append(mapping[index % 6])
        return {"pilot_rank": str(index + 1), "stable_source_image_id": f"source-{index}", "split": "train" if index % 2 else "val", "sequence_id": f"sequence-{index}", "image_path": f"images/{index}.JPG", "mask_path": f"masks/{index}.png", "annotation_status": "candidate_unreviewed", "selection_strata": "|".join(strata), "component_count": "3" if index % 7 == 0 else "1", "big_rock_pixel_count": "10", "bedrock_adjacent_pixels": "0", "geometry_status": "unresolved", "selection_seed": "42", "selection_rationale": "fixture"}

    def test_selection_is_deterministic_and_covers_development_protocol_stressors(self) -> None:
        records = [self._record(index) for index in range(40)]
        first = select_calibration_records(records, target_size=24, seed=42)
        second = select_calibration_records(records, target_size=24, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(len({record["sequence_id"] for record in first}), 24)
        labels = {label for record in first for label in record["calibration_strata"].split("|")}
        self.assertTrue({"isolated_candidate", "multiple_candidate_regions", "border_truncation_candidate", "fragmented_looking"}.issubset(labels))

    def test_expert_rows_are_rejected(self) -> None:
        records = [self._record(index) for index in range(24)]
        records[0]["split"] = "expert_min1"
        with self.assertRaisesRegex(ValueError, "only train and val"):
            select_calibration_records(records, target_size=24, seed=42)


if __name__ == "__main__":
    unittest.main()