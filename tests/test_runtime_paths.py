import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset import load_pairs_from_manifest
from src.runtime import is_kaggle_input_path, resolve_runtime_paths


class RuntimePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "ai4mars-dataset-merged-0.6"
        self.image_relative = "msl/ncam/images/example.JPG"
        self.mask_relative = "msl/ncam/labels/train/example.png"
        image_path = self.dataset_root / self.image_relative
        mask_path = self.dataset_root / self.mask_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (1, 2, 3)).save(image_path)
        Image.new("L", (8, 8), 1).save(mask_path)
        self.manifest = self.root / "split.csv"
        self._write_manifest(self.image_relative, self.mask_relative)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_manifest(self, image_path: str, mask_path: str) -> None:
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["dataset_relative_image_path", "dataset_relative_mask_path"])
            writer.writeheader()
            writer.writerow({"dataset_relative_image_path": image_path, "dataset_relative_mask_path": mask_path})

    def test_same_logical_manifest_resolves_under_different_roots(self) -> None:
        second_root = self.root / "second" / self.dataset_root.name
        for relative_path in (self.image_relative, self.mask_relative):
            source = self.dataset_root / relative_path
            target = second_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())

        first_pair = load_pairs_from_manifest(self.manifest, dataset_root=self.dataset_root)
        second_pair = load_pairs_from_manifest(self.manifest, dataset_root=second_root)

        self.assertEqual(first_pair[0][0].relative_to(self.dataset_root).as_posix(), self.image_relative)
        self.assertEqual(second_pair[0][0].relative_to(second_root).as_posix(), self.image_relative)

    def test_manifest_rejects_traversal_and_missing_files(self) -> None:
        self._write_manifest("../outside.JPG", self.mask_relative)
        with self.assertRaisesRegex(ValueError, "traverse"):
            load_pairs_from_manifest(self.manifest, dataset_root=self.dataset_root)

        self._write_manifest("msl/ncam/images/missing.JPG", self.mask_relative)
        with self.assertRaisesRegex(FileNotFoundError, "missing files"):
            load_pairs_from_manifest(self.manifest, dataset_root=self.dataset_root)

    def test_kaggle_outputs_cannot_use_input_mount(self) -> None:
        with self.assertRaisesRegex(ValueError, "Kaggle inputs"):
            resolve_runtime_paths(
                project_root=self.root,
                dataset_root=self.dataset_root,
                output_root=Path("/kaggle/input/ai4mars-output"),
            )
        self.assertTrue(is_kaggle_input_path(Path("/kaggle/input/example")))

    def test_rejects_git_bash_converted_kaggle_output_outside_kaggle(self) -> None:
        with self.assertRaisesRegex(ValueError, "Kaggle paths are valid only"):
            resolve_runtime_paths(
                project_root=self.root,
                dataset_root=self.dataset_root,
                output_root=Path("C:/Program Files/Git/kaggle/working/ai4mars"),
                environ={},
            )

    def test_explicit_paths_override_environment_and_kaggle_defaults(self) -> None:
        paths = resolve_runtime_paths(
            project_root=self.root,
            dataset_root=self.dataset_root,
            output_root=self.root / "generated",
            run_id="portable-run",
            accelerator="cpu",
            environ={"KAGGLE_KERNEL_RUN_TYPE": "Interactive", "AI4MARS_OUTPUT_ROOT": "/ignored"},
        )
        self.assertTrue(paths.kaggle)
        self.assertEqual(paths.output_root, self.root / "generated")
        self.assertEqual(paths.accelerator, "cpu")
        paths.ensure_writable_roots()
        self.assertTrue(paths.event_root.exists())


if __name__ == "__main__":
    unittest.main()