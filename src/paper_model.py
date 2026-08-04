"""Maintained DeepLabv3+ configuration for the AI4Mars paper reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PAPER_ARCHITECTURE = "DeepLabV3Plus"
PAPER_BACKBONE = "resnet101"
PAPER_PRETRAINED_WEIGHTS = "imagenet"
PAPER_INPUT_SIZE = (513, 513)
PAPER_OUTPUT_STRIDE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_MAPPING = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
IGNORE_INDEX = 255


@dataclass(frozen=True)
class DeepLabV3PlusSpec:
    architecture: str = PAPER_ARCHITECTURE
    backbone: str = PAPER_BACKBONE
    pretrained_weights: str = PAPER_PRETRAINED_WEIGHTS
    output_stride: int = PAPER_OUTPUT_STRIDE
    input_size: tuple[int, int] = PAPER_INPUT_SIZE
    num_classes: int = len(CLASS_MAPPING)
    ignore_index: int = IGNORE_INDEX
    normalization_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_STD

    def metadata(self) -> dict[str, Any]:
        return asdict(self) | {"class_mapping": CLASS_MAPPING.copy()}


def validate_deeplabv3plus_spec(spec: DeepLabV3PlusSpec) -> None:
    if spec.architecture != PAPER_ARCHITECTURE:
        raise ValueError(f"Paper reproduction requires architecture={PAPER_ARCHITECTURE}.")
    if spec.backbone != PAPER_BACKBONE:
        raise ValueError(f"Paper reproduction requires backbone={PAPER_BACKBONE}.")
    if spec.pretrained_weights != PAPER_PRETRAINED_WEIGHTS:
        raise ValueError(f"Paper reproduction requires pretrained_weights={PAPER_PRETRAINED_WEIGHTS}.")
    if spec.input_size != PAPER_INPUT_SIZE:
        raise ValueError(f"Paper reproduction requires input_size={PAPER_INPUT_SIZE}.")
    if spec.output_stride not in {8, 16}:
        raise ValueError("DeepLabv3+ output_stride must be 8 or 16.")
    if spec.num_classes != len(CLASS_MAPPING) or spec.ignore_index != IGNORE_INDEX:
        raise ValueError("Paper reproduction requires four NAV classes and ignore_index=255.")
    if len(spec.normalization_mean) != 3 or len(spec.normalization_std) != 3:
        raise ValueError("Encoder normalization must provide three channel values.")
    if any(value <= 0 for value in spec.normalization_std):
        raise ValueError("Encoder normalization standard deviations must be positive.")


def build_deeplabv3plus(spec: DeepLabV3PlusSpec):
    """Build the maintained SMP DeepLabv3+ implementation after validation."""
    validate_deeplabv3plus_spec(spec)
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3Plus(
        encoder_name=spec.backbone,
        encoder_weights=spec.pretrained_weights,
        encoder_output_stride=spec.output_stride,
        in_channels=3,
        classes=spec.num_classes,
    )