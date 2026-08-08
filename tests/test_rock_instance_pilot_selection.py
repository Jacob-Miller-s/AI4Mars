import tempfile
import unittest
from pathlib import Path

from src.rock_instance.pilot_selection import select_pilot_records, source_compatible_records, summarize_pilot


class PilotSelectionTests(unittest.TestCase):
    def _image_record(self, index: int) -> dict[str, str]:
        return {
            "stable_source_image_id": f"source-{index}", "split": "train" if index % 2 else "val",
            "sequence_id": f"sequence-{index}", "image_path": f"images/{index}.JPG", "mask_path": f"masks/{index}.png",
            "component_count": "2" if index % 5 == 0 else "1", "big_rock_pixel_count": "100",
            "bedrock_adjacent_pixels": "10" if index % 3 == 0 else "0", "has_border_component": "true" if index % 7 == 0 else "false",
            "has_very_large_component": "false", "has_unusual_aspect_ratio": "false", "has_tiny_component": "false",
            "manual_review_priority": str(index % 6),
        }

    def _geometry_record(self, index: int) -> dict[str, str]:
        return {"stable_source_image_id": f"source-{index}", "split": "train" if index % 2 else "val", "usable_for_stereo_geometry": "false", "stereo_pairing_status": "unresolved_no_explicit_pair_metadata"}

    def test_selects_deterministic_candidate_only_manifest_with_sequence_diversity(self) -> None:
        images = [self._image_record(index) for index in range(220)]
        geometry = [self._geometry_record(index) for index in range(220)]

        first = select_pilot_records(images, geometry, target_size=100, seed=42)
        second = select_pilot_records(images, geometry, target_size=100, seed=42)
        summary = summarize_pilot(first, target_size=100, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertTrue(all(row["annotation_status"] == "candidate_unreviewed" for row in first))
        self.assertEqual(summary["unique_sequences"], 100)
        self.assertTrue(all("rgb_only_or_geometry_unresolved" in row["selection_strata"] for row in first))
        self.assertGreater(summary["strata_counts"]["isolated_candidate"], 0)
        self.assertGreater(summary["strata_counts"]["multiple_candidate_regions"], 0)

    def test_rejects_non_pilot_target_sizes(self) -> None:
        images = [self._image_record(index) for index in range(220)]
        geometry = [self._geometry_record(index) for index in range(220)]
        with self.assertRaisesRegex(ValueError, "between 100 and 200"):
            select_pilot_records(images, geometry, target_size=99, seed=42)

    def test_partitions_source_rows_without_silently_dropping_them(self) -> None:
        records = [self._image_record(1), self._image_record(2)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            for relative_path in (records[0]["image_path"], records[0]["mask_path"]):
                source_path = dataset_root / relative_path
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.touch()

            compatible, unavailable = source_compatible_records(records, dataset_root)

        self.assertEqual(compatible, [records[0]])
        self.assertEqual(
            unavailable,
            [{
                "stable_source_image_id": "source-2",
                "image_path": "images/2.JPG",
                "mask_path": "masks/2.png",
                "missing_sources": "image|mask",
            }],
        )


if __name__ == "__main__":
    unittest.main()