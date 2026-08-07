import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.rock_instance.common import load_development_manifest_rows, require_development_splits
from src.rock_instance.stereo_inventory import inventory_records, summarize_records


class StereoInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "dataset"
        self.manifest_root = self.root / "manifests"
        self.image_relative_path = "msl/ncam/images/edr/NLA_1EDR_F001NCAM00001M1.JPG"
        image_path = self.dataset_root / self.image_relative_path
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"image")
        range_path = self.dataset_root / "msl/ncam/images/rng-30m/NLA_1RNG_F001NCAM00001M1.png"
        range_path.parent.mkdir(parents=True)
        range_path.write_bytes(b"mask")
        self._write_manifest("train.csv", [self._row("A", "SEQ_A")])
        self._write_manifest("val.csv", [self._row("B", "SEQ_B")])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _row(self, source_id: str, sequence_id: str) -> dict[str, str]:
        return {
            "dataset_relative_image_path": self.image_relative_path,
            "dataset_relative_mask_path": f"msl/ncam/labels/train/{source_id}.png",
            "stable_source_image_id": source_id,
            "sequence_id": sequence_id,
            "mission": "msl",
            "rover": "curiosity",
            "camera": "ncam",
            "label_scheme": "NAV",
            "label_role": "crowdsourced_train",
            "agreement_threshold": "",
            "image_width": "8",
            "image_height": "8",
            "mask_width": "8",
            "mask_height": "8",
            "per_class_pixel_counts_json": json.dumps({"0": 64, "1": 0, "2": 0, "3": 0, "255": 0}),
        }

    def _write_manifest(self, name: str, rows: list[dict[str, str]]) -> None:
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        with (self.manifest_root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_inventory_records_range_validity_mask_without_claiming_depth_or_stereo(self) -> None:
        rows = load_development_manifest_rows(self.manifest_root, {"train": "train.csv", "val": "val.csv"}, ["train", "val"])

        records = inventory_records(self.dataset_root, rows)
        summary = summarize_records(records, ("train", "val"))

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["range_validity_mask_availability"], "present")
        self.assertEqual(records[0]["range_depth_availability"], "unresolved_no_depth_product_metadata")
        self.assertEqual(records[0]["stereo_pairing_status"], "unresolved_no_explicit_pair_metadata")
        self.assertFalse(records[0]["usable_for_stereo_geometry"])
        self.assertEqual(summary["confirmed_stereo_mates"], 0)
        self.assertEqual(summary["range_validity_masks_present"], 2)

    def test_expert_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only train and val"):
            require_development_splits(["train", "expert_min1"])


if __name__ == "__main__":
    unittest.main()