import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ai4mars.dataset import AI4MarsDataset, find_shape_mismatches, load_pairs_from_manifest


class DatasetManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_image(self, path: Path, size=(8, 8), mode="RGB") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        color = (64, 128, 192) if mode == "RGB" else 1
        Image.new(mode, size, color).save(path)

    def _write_manifest(self, rows):
        manifest = self.root / "pairs.csv"
        with manifest.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["selected_image_path", "mask_path", "shape_match", "label_scheme"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_find_shape_mismatches_detects_pairs(self) -> None:
        image_ok = self.root / "img_ok.jpg"
        mask_ok = self.root / "mask_ok.png"
        image_bad = self.root / "img_bad.jpg"
        mask_bad = self.root / "mask_bad.png"
        self._write_image(image_ok, size=(8, 8), mode="RGB")
        self._write_image(mask_ok, size=(8, 8), mode="L")
        self._write_image(image_bad, size=(8, 8), mode="RGB")
        self._write_image(mask_bad, size=(10, 10), mode="L")

        mismatches = find_shape_mismatches([(image_ok, mask_ok), (image_bad, mask_bad)])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0][0], image_bad)
        self.assertEqual(mismatches[0][1], mask_bad)

    def test_load_pairs_from_manifest_filters_shape_mismatch_rows(self) -> None:
        image_ok = self.root / "img_ok.jpg"
        mask_ok = self.root / "mask_ok.png"
        image_bad = self.root / "img_bad.jpg"
        mask_bad = self.root / "mask_bad.png"
        self._write_image(image_ok, size=(8, 8), mode="RGB")
        self._write_image(mask_ok, size=(8, 8), mode="L")
        self._write_image(image_bad, size=(8, 8), mode="RGB")
        self._write_image(mask_bad, size=(10, 10), mode="L")

        manifest = self._write_manifest(
            [
                {
                    "selected_image_path": str(image_ok),
                    "mask_path": str(mask_ok),
                    "shape_match": "1",
                    "label_scheme": "NAV",
                },
                {
                    "selected_image_path": str(image_bad),
                    "mask_path": str(mask_bad),
                    "shape_match": "0",
                    "label_scheme": "NAV",
                },
            ]
        )

        pairs = load_pairs_from_manifest(manifest, required_label_scheme="NAV")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], (image_ok, mask_ok))

    def test_load_pairs_from_manifest_raises_on_geometry_mismatch_without_shape_flag(self) -> None:
        image_bad = self.root / "img_bad.jpg"
        mask_bad = self.root / "mask_bad.png"
        self._write_image(image_bad, size=(8, 8), mode="RGB")
        self._write_image(mask_bad, size=(10, 10), mode="L")

        manifest = self.root / "pairs_no_shape_column.csv"
        with manifest.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["selected_image_path", "mask_path"])
            writer.writeheader()
            writer.writerow(
                {
                    "selected_image_path": str(image_bad),
                    "mask_path": str(mask_bad),
                }
            )

        with self.assertRaises(ValueError):
            load_pairs_from_manifest(manifest, require_shape_match=True)

    def test_dataset_strict_shape_match_raises(self) -> None:
        image_bad = self.root / "img_bad.jpg"
        mask_bad = self.root / "mask_bad.png"
        self._write_image(image_bad, size=(8, 8), mode="RGB")
        self._write_image(mask_bad, size=(10, 10), mode="L")

        with self.assertRaises(ValueError):
            AI4MarsDataset([(image_bad, mask_bad)], require_original_shape_match=True)


if __name__ == "__main__":
    unittest.main()
