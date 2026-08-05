"""Maintained DeepLabv3+ configuration for the AI4Mars paper reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PaperAlignedDeepLabV3Plus(nn.Module):
    """Preserves the paper's 513x513 evaluation domain around SMP's stride constraint.

    ``segmentation_models_pytorch``'s ``DeepLabV3Plus`` rejects any input whose
    height/width is not divisible by its ``encoder_output_stride`` (16 for this
    reproduction), but the AI4Mars paper resizes images to 513x513, which is not
    divisible by 16. Rather than change the paper-aligned resolution, this
    adapter pads the already-normalized input up to the next multiple of
    ``padding_multiple`` (513 -> 528), runs the unmodified SMP network, and crops
    the returned logits back down to the original height/width before they are
    ever compared against a (never-padded) target mask. Padding uses constant
    value 0.0 in normalized space, which corresponds to the per-channel ImageNet
    encoder mean, not raw black. The 528x528 shape is purely an internal
    implementation/compatibility detail: it must never be reported as the
    experimental input resolution, and the padded border must never reach loss,
    confusion matrices, or metrics.
    """

    def __init__(self, network: nn.Module, padding_multiple: int) -> None:
        super().__init__()
        self.network = network
        self.padding_multiple = padding_multiple

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_height, original_width = inputs.shape[-2:]

        pad_height = (-original_height) % self.padding_multiple
        pad_width = (-original_width) % self.padding_multiple

        if pad_height or pad_width:
            inputs = F.pad(
                inputs,
                (0, pad_width, 0, pad_height),
                mode="constant",
                value=0.0,
            )

        logits = self.network(inputs)

        return logits[
            ...,
            :original_height,
            :original_width,
        ]


def paper_padding_metadata(spec: DeepLabV3PlusSpec) -> dict[str, Any]:
    """Describe the adapter's padding/cropping behavior for ``spec.input_size``.

    These keys record an internal implementation/compatibility detail; the
    scientifically reported model input resolution must remain
    ``spec.input_size`` (e.g. 513x513), never the padded internal size.
    """
    height, width = spec.input_size
    multiple = spec.output_stride
    padded_height = height + ((-height) % multiple)
    padded_width = width + ((-width) % multiple)
    return {
        "requested_input_size": [height, width],
        "internal_padding_multiple": multiple,
        "internal_padded_size_for_513": [padded_height, padded_width],
        "input_padding_policy": "right_bottom",
        "input_padding_mode": "constant",
        "normalized_padding_value": 0.0,
        "output_crop_policy": "original_spatial_extent",
    }


def build_deeplabv3plus(spec: DeepLabV3PlusSpec) -> PaperAlignedDeepLabV3Plus:
    """Build the maintained SMP DeepLabv3+ implementation after validation.

    The returned model is always wrapped in ``PaperAlignedDeepLabV3Plus`` so
    every caller transparently receives 513x513-in/513x513-out behavior; no
    caller should construct ``smp.DeepLabV3Plus`` directly.
    """
    validate_deeplabv3plus_spec(spec)
    import segmentation_models_pytorch as smp

    network = smp.DeepLabV3Plus(
        encoder_name=spec.backbone,
        encoder_weights=spec.pretrained_weights,
        encoder_output_stride=spec.output_stride,
        in_channels=3,
        classes=spec.num_classes,
    )
    return PaperAlignedDeepLabV3Plus(network=network, padding_multiple=spec.output_stride)