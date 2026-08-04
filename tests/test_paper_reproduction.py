import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import AI4MarsDataset
from src.paper_model import DeepLabV3PlusSpec, IMAGENET_MEAN, IMAGENET_STD, validate_deeplabv3plus_spec
from src.paper_train import load_and_validate_config
from src.paper_reproduction import (
    assert_no_reproduction_leakage,
    compute_paper_class_composition,
    summarize_reproduction_manifests,
    validate_reproduction_manifest,
)
from src.train_utils import evaluate, train_one_epoch


class PaperReproductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _row(self, source_id: str, sequence_id: str, *, agreement: str = "", camera: str = "ncam") -> dict[str, str]:
        return {
            "dataset_relative_image_path": f"msl/{camera}/images/{source_id}.JPG",
            "dataset_relative_mask_path": f"msl/{camera}/labels/{source_id}.png",
            "stable_source_image_id": source_id,
            "sequence_id": sequence_id,
            "mission": "msl",
            "rover": "curiosity",
            "camera": camera,
            "label_scheme": "NAV",
            "label_role": "expert_gold_test" if agreement else "crowdsourced_train",
            "agreement_threshold": agreement,
            "image_width": "8",
            "image_height": "8",
            "mask_width": "8",
            "mask_height": "8",
            "per_class_pixel_counts_json": json.dumps({"0": 50, "1": 25, "2": 15, "3": 10, "255": 20}),
        }

    def _manifest(self, name: str, rows: list[dict[str, str]]) -> Path:
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_paper_weighting_uses_complement_of_labeled_composition(self) -> None:
        manifest = self._manifest("train.csv", [self._row("A", "SEQ_A")])

        composition = compute_paper_class_composition(manifest)

        self.assertEqual(composition.pixel_counts, (50, 25, 15, 10))
        self.assertEqual(composition.ignore_pixel_count, 20)
        self.assertEqual(composition.valid_pixel_count, 100)
        self.assertEqual(composition.class_proportions, (0.5, 0.25, 0.15, 0.1))
        self.assertEqual(composition.class_weights, (0.5, 0.75, 0.85, 0.9))
        self.assertEqual(len(composition.manifest_sha256), 64)

    def test_scope_validation_rejects_mastcam(self) -> None:
        manifest = self._manifest("train.csv", [self._row("A", "SEQ_A", camera="mcam")])

        with self.assertRaisesRegex(ValueError, "MSL NavCam"):
            validate_reproduction_manifest(manifest, split_name="train")

    def test_portability_validation_rejects_absolute_windows_path(self) -> None:
        row = self._row("A", "SEQ_A")
        row["dataset_relative_image_path"] = "C:\\dataset\\image.JPG"
        manifest = self._manifest("train.csv", [row])

        with self.assertRaisesRegex(ValueError, "non-portable"):
            validate_reproduction_manifest(manifest, split_name="train")

    def test_leakage_validation_rejects_expert_source_overlap(self) -> None:
        train = self._manifest("train.csv", [self._row("A", "SEQ_A")])
        val = self._manifest("val.csv", [self._row("B", "SEQ_B")])
        expert = self._manifest("expert.csv", [self._row("A", "SEQ_C", agreement="min3-100agree")])

        with self.assertRaisesRegex(ValueError, "source leakage"):
            assert_no_reproduction_leakage({"train": train, "val": val, "expert_min3_100agree": expert})

    def test_summary_reports_scope_role_and_agreement_counts(self) -> None:
        train = self._manifest("train.csv", [self._row("A", "SEQ_A")])
        expert = self._manifest("expert.csv", [self._row("B", "SEQ_B", agreement="min1-100agree")])

        summary = summarize_reproduction_manifests({"train": train, "expert_min1_100agree": expert})

        self.assertEqual(summary["train"]["camera:ncam"], 1)
        self.assertEqual(summary["train"]["agreement:none"], 1)
        self.assertEqual(summary["expert_min1_100agree"]["label_role:expert_gold_test"], 1)
        self.assertEqual(summary["expert_min1_100agree"]["agreement:min1-100agree"], 1)

    def test_dataset_applies_encoder_normalization_and_nearest_mask_resize(self) -> None:
        image_path = self.root / "image.png"
        mask_path = self.root / "mask.png"
        Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
        Image.fromarray(np.array([[0, 3], [255, 1]], dtype=np.uint8)).save(mask_path)
        dataset = AI4MarsDataset(
            [(image_path, mask_path)],
            image_size=(4, 4),
            normalization_mean=IMAGENET_MEAN,
            normalization_std=IMAGENET_STD,
        )

        image, mask = dataset[0]

        self.assertAlmostEqual(image[0, 0, 0].item(), (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], places=4)
        self.assertAlmostEqual(image[1, 0, 0].item(), (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], places=4)
        self.assertEqual(set(mask.flatten().tolist()), {0, 1, 3, 255})
        self.assertEqual(tuple(mask.shape), (4, 4))

    def test_model_spec_enforces_paper_architecture_contract(self) -> None:
        validate_deeplabv3plus_spec(DeepLabV3PlusSpec())

        with self.assertRaisesRegex(ValueError, "backbone"):
            validate_deeplabv3plus_spec(DeepLabV3PlusSpec(backbone="resnet34"))

    def test_gradient_accumulation_uses_expected_number_of_optimizer_steps(self) -> None:
        class CountingSgd(torch.optim.SGD):
            def __init__(self, parameters):
                super().__init__(parameters, lr=0.1)
                self.step_count = 0

            def step(self, closure=None):
                self.step_count += 1
                return super().step(closure)

        model = nn.Conv2d(1, 2, kernel_size=1)
        optimizer = CountingSgd(model.parameters())
        loader = DataLoader(
            TensorDataset(torch.ones(3, 1, 2, 2), torch.zeros(3, 2, 2, dtype=torch.long)),
            batch_size=1,
        )

        train_one_epoch(
            model,
            loader,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            gradient_accumulation_steps=2,
        )

        self.assertEqual(optimizer.step_count, 2)

    def test_evaluation_reports_weighted_and_unweighted_loss(self) -> None:
        class ConstantModel(nn.Module):
            def forward(self, images):
                return torch.zeros((images.shape[0], 4, 2, 2), dtype=images.dtype)

        loader = DataLoader(
            TensorDataset(torch.ones(1, 3, 2, 2), torch.tensor([[[0, 0], [1, 1]]])),
            batch_size=1,
        )

        metrics = evaluate(
            ConstantModel(),
            loader,
            nn.CrossEntropyLoss(weight=torch.tensor([4.0, 1.0, 1.0, 1.0])),
            torch.device("cpu"),
            return_detailed_metrics=True,
            unweighted_loss_fn=nn.CrossEntropyLoss(),
        )

        self.assertIn("unweighted_val_loss", metrics)
        self.assertAlmostEqual(metrics["unweighted_val_loss"], float(torch.log(torch.tensor(4.0))), places=6)

    def test_configuration_validation_rejects_nonpaper_backbone(self) -> None:
        config = self.root / "config.yaml"
        config.write_text(
            """runtime: {}\ndata:\n  dataset_manifest: manifest.csv\n  train_manifest: train.csv\n  val_manifest: val.csv\n  expert_min1_manifest: min1.csv\n  expert_min2_manifest: min2.csv\n  expert_min3_manifest: min3.csv\nmodel:\n  architecture: DeepLabV3Plus\n  backbone: resnet34\n  pretrained_weights: imagenet\n  input_size: [513, 513]\ntraining:\n  ignore_index: 255\n  class_weighting: paper_complement_composition\n  optimizer: adamw\n  scheduler: none\n  batch_size: 1\n  gradient_accumulation_steps: 1\n  epochs: 1\n  checkpoint_interval: 1\n  validation_interval: 1\n  batch_log_interval: 1\n  learning_rate: 0.0001\n  weight_decay: 0.0\nlogging: {}\n""",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "backbone"):
            load_and_validate_config(config)


if __name__ == "__main__":
    unittest.main()