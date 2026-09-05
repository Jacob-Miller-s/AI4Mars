import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ai4mars.paper_train import _preflight_zero_valid_training_rows


class PreflightZeroValidFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset_root = self.root / "ai4mars-dataset-merged-0.6"
        self.images_dir = self.dataset_root / "msl" / "ncam" / "images"
        self.labels_dir = self.dataset_root / "msl" / "ncam" / "labels"
        self.images_dir.mkdir(parents=True)
        self.labels_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_pair(self, stem: str, mask: np.ndarray) -> None:
        Image.new("RGB", (mask.shape[1], mask.shape[0]), color=(20, 30, 40)).save(self.images_dir / f"{stem}.JPG")
        Image.fromarray(mask.astype(np.uint8)).save(self.labels_dir / f"{stem}.png")

    def _row(self, stem: str, counts: dict[str, int], sequence_id: str) -> dict[str, str]:
        return {
            "dataset_relative_image_path": f"msl/ncam/images/{stem}.JPG",
            "dataset_relative_mask_path": f"msl/ncam/labels/{stem}.png",
            "stable_source_image_id": stem,
            "sequence_id": sequence_id,
            "mission": "msl",
            "rover": "curiosity",
            "camera": "ncam",
            "label_scheme": "NAV",
            "label_role": "crowdsourced_train",
            "agreement_threshold": "",
            "image_width": "4",
            "image_height": "4",
            "mask_width": "4",
            "mask_height": "4",
            "per_class_pixel_counts_json": json.dumps(counts, sort_keys=True),
        }

    def _write_manifest(self, rows: list[dict[str, str]]) -> Path:
        manifest_path = self.root / "train.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def test_excludes_zero_valid_rows_deterministically(self) -> None:
        self._write_pair("A", np.zeros((4, 4), dtype=np.uint8))
        self._write_pair("B", np.full((4, 4), 255, dtype=np.uint8))
        self._write_pair("C", np.ones((4, 4), dtype=np.uint8))

        rows = [
            self._row("A", {"0": 16}, "SEQ_A"),
            self._row("B", {"255": 16}, "SEQ_B"),
            self._row("C", {"1": 16}, "SEQ_C"),
        ]
        manifest = self._write_manifest(rows)

        artifact_one, excluded_one, source_map_one = _preflight_zero_valid_training_rows(
            manifest,
            self.dataset_root,
            full_disk_audit=False,
        )
        artifact_two, excluded_two, source_map_two = _preflight_zero_valid_training_rows(
            manifest,
            self.dataset_root,
            full_disk_audit=False,
        )

        self.assertEqual(excluded_one, {"msl/ncam/labels/B.png"})
        self.assertEqual(excluded_one, excluded_two)
        self.assertEqual(source_map_one, source_map_two)
        self.assertEqual(artifact_one["excluded_zero_valid_rows"], 1)
        self.assertEqual(artifact_one["retained_training_rows"], 2)
        self.assertEqual(artifact_one["audit_hash"], artifact_two["audit_hash"])
        self.assertEqual(
            artifact_one["excluded_rows"][0]["stable_source_image_id"],
            "B",
        )

    def test_optional_full_disk_audit_detects_manifest_count_mismatch(self) -> None:
        self._write_pair("A", np.zeros((4, 4), dtype=np.uint8))
        self._write_pair("B", np.full((4, 4), 255, dtype=np.uint8))

        rows = [
            self._row("A", {"0": 16}, "SEQ_A"),
            self._row("B", {"255": 15}, "SEQ_B"),
        ]
        manifest = self._write_manifest(rows)

        artifact, excluded, _ = _preflight_zero_valid_training_rows(
            manifest,
            self.dataset_root,
            full_disk_audit=True,
        )

        self.assertEqual(excluded, {"msl/ncam/labels/B.png"})
        self.assertTrue(artifact["full_disk_count_audit_performed"])
        self.assertEqual(len(artifact["disk_count_mismatch_rows"]), 1)
        self.assertEqual(
            artifact["disk_count_mismatch_rows"][0]["stable_source_image_id"],
            "B",
        )


if __name__ == "__main__":
    unittest.main()
