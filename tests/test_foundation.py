import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ai4mars.dataset import load_pairs_from_manifest
from ai4mars.foundation import build_split_manifests, write_dataset_manifest, write_run_record, _row_matches_scope


class DatasetScopeFilterTests(unittest.TestCase):
    def setUp(self):
        self.msl_navcam_row = {
            "mission": "msl",
            "camera": "ncam",
            "Label_scheme": "NAV",
        }

        self.msl_navcam_row = {
            "mission": "msl",
            "camera": "ncam",
            "Label_scheme": "NAV",
        }

        self.mer_navcam_row = {
            "mission": "mer",
            "camera": "navcam",
            "Label_scheme": "NAV",
        }

        def test_matching_msl_navcam_row_in_scope(self):
            result = _row_matches_scope(self.msl_navcam_row, mission_filter="msl", camera_filter="ncam")

            self.assertTrue(result)

        def test_non_matching_mission_row_out_of_scope(self):
            result = _row_matches_scope(self.mer_navcam_row, mission_filter="msl", camera_filter="ncam")

            self.assertFalse(result)

        def test_none_filters(self):
            result = _row_matches_scope(self.msl_navcam_row, mission_filter=None, camera_filter=None)

            self.assertTrue(result)

class FoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "ai4mars-dataset-merged-0.6"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_image(self, rel_path: str, size=(8, 8), mode="RGB", color=64) -> Path:
        path = self.dataset_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "RGB":
            pixel = (color, color, color)
        else:
            pixel = color
        Image.new(mode, size, pixel).save(path)
        return path

    def _seed_dataset(self) -> None:
        self._write_image(
            "msl/ncam/images/edr/NLA_397586934EDR_F0010008AUT_04096M1.JPG",
            mode="RGB",
        )
        self._write_image(
            "msl/ncam/labels/train/NLA_397586934EDR_F0010008AUT_04096M1.png",
            mode="L",
            color=1,
        )

        self._write_image(
            "msl/ncam/images/edr/NLB_565812047EDR_F0671016NCAM00312M1.JPG",
            mode="RGB",
        )
        self._write_image(
            "msl/ncam/labels/train/NLB_565812047EDR_F0671016NCAM00312M1.png",
            mode="L",
            color=2,
        )

        self._write_image(
            "msl/ncam/images/edr/NLA_409036068EDR_F0051606NCAM00348M1.JPG",
            mode="RGB",
        )
        for threshold, color in (("min1-100agree", 1), ("min2-100agree", 2), ("min3-100agree", 3)):
            self._write_image(
                f"msl/ncam/labels/test/masked-gold-{threshold}/NLA_409036068EDR_F0051606NCAM00348M1_merged.png",
                mode="L",
                color=color,
            )

        self._write_image("mer/images/eff/2F999999999EFFA2F1000L0M1.JPG", mode="RGB")
        self._write_image(
            "mer/labels/train/2F000000000EFFA2F1000L0M1_merged6.png",
            mode="L",
            color=1,
        )

    def test_write_dataset_manifest_enumerates_exclusions_and_metadata(self) -> None:
        self._seed_dataset()
        manifest_path = self.root / "dataset_manifest.csv"

        rows = write_dataset_manifest(self.dataset_root, manifest_path)

        matched_row = next(
            row for row in rows if row["dataset_relative_mask_path"] == "msl/ncam/labels/train/NLA_397586934EDR_F0010008AUT_04096M1.png"
        )
        self.assertEqual(matched_row["dataset_version"], "ai4mars-dataset-merged-0.6")
        self.assertEqual(matched_row["dataset_doi"], "10.5281/zenodo.15995036")
        self.assertEqual(matched_row["archive_md5"], "daf80a86021253292e6c425f97baa5c6")
        self.assertEqual(matched_row["label_role"], "crowdsourced_train")
        self.assertEqual(matched_row["label_scheme"], "NAV")
        self.assertEqual(matched_row["exclusion_reason"], "")
        self.assertTrue(float(matched_row["valid_pixel_fraction"]) > 0.0)
        self.assertIn('"1":', matched_row["per_class_pixel_counts_json"])

        unmatched_mask_row = next(
            row for row in rows if row["dataset_relative_mask_path"] == "mer/labels/train/2F000000000EFFA2F1000L0M1_merged6.png"
        )
        self.assertEqual(unmatched_mask_row["exclusion_reason"], "unmatched_mask_no_candidate_image")

        unmatched_image_row = next(
            row for row in rows if row["dataset_relative_image_path"] == "mer/images/eff/2F999999999EFFA2F1000L0M1.JPG"
        )
        self.assertEqual(unmatched_image_row["exclusion_reason"], "unmatched_image_unused")

    def test_build_split_manifests_preserves_expert_threshold_variants(self) -> None:
        self._seed_dataset()
        dataset_manifest_path = self.root / "dataset_manifest.csv"
        write_dataset_manifest(self.dataset_root, dataset_manifest_path)

        split_paths = build_split_manifests(dataset_manifest_path, self.root / "splits", train_ratio=0.5, seed=7)

        self.assertIn("train", split_paths)
        self.assertIn("val", split_paths)
        self.assertIn("test_min1_100agree", split_paths)
        self.assertIn("test_min2_100agree", split_paths)
        self.assertIn("test_min3_100agree", split_paths)

        train_pairs = load_pairs_from_manifest(split_paths["train"], dataset_root=self.dataset_root, required_label_scheme="NAV")
        val_pairs = load_pairs_from_manifest(split_paths["val"], dataset_root=self.dataset_root, required_label_scheme="NAV")
        min1_pairs = load_pairs_from_manifest(split_paths["test_min1_100agree"], dataset_root=self.dataset_root, required_label_scheme="NAV")

        self.assertEqual(len(train_pairs), 1)
        self.assertEqual(len(val_pairs), 1)
        self.assertEqual(len(min1_pairs), 1)

        train_ids = {pair[0].stem for pair in train_pairs}
        val_ids = {pair[0].stem for pair in val_pairs}
        self.assertFalse(train_ids & val_ids)
        self.assertNotIn("NLA_409036068EDR_F0051606NCAM00348M1", train_ids | val_ids)

    def test_build_split_manifests_excludes_train_rows_with_expert_sequence_overlap(self) -> None:
        self._seed_dataset()
        self._write_image(
            "msl/ncam/images/edr/NLA_500000000EDR_F0051606NCAM00348M1.JPG",
            mode="RGB",
        )
        self._write_image(
            "msl/ncam/labels/train/NLA_500000000EDR_F0051606NCAM00348M1.png",
            mode="L",
            color=1,
        )

        dataset_manifest_path = self.root / "dataset_manifest.csv"
        write_dataset_manifest(self.dataset_root, dataset_manifest_path)
        split_paths = build_split_manifests(dataset_manifest_path, self.root / "splits", train_ratio=0.5, seed=7)

        train_pairs = load_pairs_from_manifest(split_paths["train"], dataset_root=self.dataset_root, required_label_scheme="NAV")
        val_pairs = load_pairs_from_manifest(split_paths["val"], dataset_root=self.dataset_root, required_label_scheme="NAV")
        all_internal_ids = {pair[0].stem for pair in train_pairs} | {pair[0].stem for pair in val_pairs}
        self.assertNotIn("NLA_500000000EDR_F0051606NCAM00348M1", all_internal_ids)

    def test_write_run_record_outputs_required_files(self) -> None:
        self._seed_dataset()
        dataset_manifest_path = self.root / "dataset_manifest.csv"
        write_dataset_manifest(self.dataset_root, dataset_manifest_path)
        split_paths = build_split_manifests(dataset_manifest_path, self.root / "splits", train_ratio=0.5, seed=7)

        run_dir = self.root / "artifacts" / "runs" / "demo-run"
        write_run_record(
            run_dir,
            config={"seed": 7, "model": "demo"},
            dataset_manifest_path=dataset_manifest_path,
            split_manifest_paths=split_paths,
            metrics={"miou": 0.5},
            environment_text="python=test\n",
        )

        self.assertTrue((run_dir / "config.json").exists())
        self.assertTrue((run_dir / "dataset_manifest_hash.txt").exists())
        self.assertTrue((run_dir / "split_manifest_hashes.json").exists())
        self.assertTrue((run_dir / "metrics.json").exists())
        self.assertTrue((run_dir / "environment.txt").exists())
        self.assertTrue((run_dir / "figures").exists())

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["miou"], 0.5)


if __name__ == "__main__":
    unittest.main()