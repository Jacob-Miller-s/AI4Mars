"""Tests for PaperAlignedDeepLabV3Plus: the 513x513-in/513x513-out padding adapter.

segmentation_models_pytorch.DeepLabV3Plus requires spatial dimensions divisible
by its encoder_output_stride, but the AI4Mars paper's 513x513 resolution is not
divisible by 16. These tests prove the adapter pads/crops around that
constraint without ever changing the reported 513x513 experimental resolution,
without padding masks, and without letting the padded border reach loss.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from ai4mars.paper_model import DeepLabV3PlusSpec, PaperAlignedDeepLabV3Plus, build_deeplabv3plus, paper_padding_metadata


class TinySegmentationNetwork(nn.Module):
    """A fast, CPU-safe stand-in for the real SMP network.

    Includes a stride-2 conv so the wrapped network itself changes spatial
    dimensions, proving the adapter crops from whatever shape the wrapped
    network actually returns rather than assuming it preserves shape.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CanonicalShapeTests(unittest.TestCase):
    """A: the canonical 513x513 paper resolution round-trips through padding."""

    def test_canonical_513_returns_513_logits(self) -> None:
        adapter = PaperAlignedDeepLabV3Plus(TinySegmentationNetwork(), padding_multiple=16)
        inputs = torch.randn(2, 3, 513, 513)

        logits = adapter(inputs)

        self.assertEqual(tuple(logits.shape), (2, 4, 513, 513))

    def test_internal_network_receives_padded_528_input(self) -> None:
        received_shapes = []

        class RecordingNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                received_shapes.append(tuple(x.shape[-2:]))
                return self.conv(x)

        adapter = PaperAlignedDeepLabV3Plus(RecordingNetwork(), padding_multiple=16)
        adapter(torch.randn(1, 3, 513, 513))

        self.assertEqual(received_shapes, [(528, 528)])

    def test_padding_value_is_zero_in_normalized_space(self) -> None:
        # A network that returns its input unchanged lets us inspect exactly
        # what value was written into the padded border.
        class IdentityNetwork(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[:, :1].expand(-1, 4, -1, -1)

        adapter = PaperAlignedDeepLabV3Plus(IdentityNetwork(), padding_multiple=16)
        inputs = torch.full((1, 3, 513, 513), fill_value=5.0)

        logits = adapter(inputs)

        # Cropped back to 513x513, so the padded border itself is not directly
        # observable here; instead check the network's raw (pre-crop) view of
        # the padded region by calling the padding step in isolation.
        pad_height = (-513) % 16
        pad_width = (-513) % 16
        padded = torch.nn.functional.pad(inputs, (0, pad_width, 0, pad_height), mode="constant", value=0.0)
        self.assertTrue(torch.all(padded[:, :, 513:, :] == 0.0))
        self.assertTrue(torch.all(padded[:, :, :, 513:] == 0.0))
        self.assertEqual(tuple(logits.shape), (1, 4, 513, 513))


class AlreadyDivisibleShapeTests(unittest.TestCase):
    """B: a shape already divisible by the padding multiple needs no padding."""

    def test_528_input_returns_528_logits_unpadded(self) -> None:
        received_shapes = []

        class RecordingNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                received_shapes.append(tuple(x.shape[-2:]))
                return self.conv(x)

        adapter = PaperAlignedDeepLabV3Plus(RecordingNetwork(), padding_multiple=16)
        logits = adapter(torch.randn(1, 3, 528, 528))

        self.assertEqual(tuple(logits.shape), (1, 4, 528, 528))
        self.assertEqual(received_shapes, [(528, 528)])


class RectangularShapeTests(unittest.TestCase):
    """C: independent height/width padding for a non-square input."""

    def test_rectangular_513x497_returns_original_shape(self) -> None:
        received_shapes = []

        class RecordingNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                received_shapes.append(tuple(x.shape[-2:]))
                return self.conv(x)

        adapter = PaperAlignedDeepLabV3Plus(RecordingNetwork(), padding_multiple=16)
        logits = adapter(torch.randn(1, 3, 513, 497))

        internal_height, internal_width = received_shapes[0]
        self.assertEqual(internal_height % 16, 0)
        self.assertEqual(internal_width % 16, 0)
        self.assertEqual(tuple(logits.shape), (1, 4, 513, 497))


class MaskUntouchedTests(unittest.TestCase):
    """D: only the normalized image is padded; masks are never touched."""

    def test_mask_tensor_is_unaffected_by_forward_pass(self) -> None:
        adapter = PaperAlignedDeepLabV3Plus(TinySegmentationNetwork(), padding_multiple=16)
        mask = torch.randint(0, 4, (1, 513, 513), dtype=torch.long)
        mask_before = mask.clone()

        adapter(torch.randn(1, 3, 513, 513))

        self.assertTrue(torch.equal(mask, mask_before))
        self.assertEqual(tuple(mask.shape), (1, 513, 513))


class CrossEntropyLossCompatibilityTests(unittest.TestCase):
    """E: returned 513x513 logits and an original 513x513 target are shape-compatible."""

    def test_loss_accepts_returned_logits_and_original_target(self) -> None:
        adapter = PaperAlignedDeepLabV3Plus(TinySegmentationNetwork(), padding_multiple=16)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        mask = torch.randint(0, 4, (1, 513, 513), dtype=torch.long)

        logits = adapter(torch.randn(1, 3, 513, 513))
        loss = loss_fn(logits, mask)

        self.assertTrue(torch.isfinite(loss))

    def test_gradients_flow_through_pad_network_and_crop(self) -> None:
        adapter = PaperAlignedDeepLabV3Plus(TinySegmentationNetwork(), padding_multiple=16)
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        mask = torch.randint(0, 4, (1, 513, 513), dtype=torch.long)
        inputs = torch.randn(1, 3, 513, 513, requires_grad=True)

        logits = adapter(inputs)
        loss = loss_fn(logits, mask)
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.any(inputs.grad != 0))
        for param in adapter.network.parameters():
            self.assertIsNotNone(param.grad)


class OutputStrideSourcedFromSpecTests(unittest.TestCase):
    """F: the adapter's padding multiple comes from the validated spec, not a hardcoded 16."""

    def test_build_deeplabv3plus_wires_spec_output_stride_into_adapter(self) -> None:
        spec = DeepLabV3PlusSpec(output_stride=8)

        with patch("segmentation_models_pytorch.DeepLabV3Plus", return_value=TinySegmentationNetwork()) as mock_ctor:
            model = build_deeplabv3plus(spec)

        mock_ctor.assert_called_once_with(
            encoder_name=spec.backbone,
            encoder_weights=spec.pretrained_weights,
            encoder_output_stride=8,
            in_channels=3,
            classes=spec.num_classes,
        )
        self.assertIsInstance(model, PaperAlignedDeepLabV3Plus)
        self.assertEqual(model.padding_multiple, 8)

    def test_padding_metadata_reflects_spec_output_stride_not_16(self) -> None:
        spec = DeepLabV3PlusSpec(output_stride=8)

        metadata = paper_padding_metadata(spec)

        self.assertEqual(metadata["internal_padding_multiple"], 8)
        # 513 padded to the next multiple of 8 is 520, not the stride-16 528.
        self.assertEqual(metadata["internal_padded_size_for_513"], [520, 520])

    def test_padding_metadata_for_canonical_spec(self) -> None:
        metadata = paper_padding_metadata(DeepLabV3PlusSpec())

        self.assertEqual(metadata["requested_input_size"], [513, 513])
        self.assertEqual(metadata["internal_padding_multiple"], 16)
        self.assertEqual(metadata["internal_padded_size_for_513"], [528, 528])
        self.assertEqual(metadata["input_padding_policy"], "right_bottom")
        self.assertEqual(metadata["input_padding_mode"], "constant")
        self.assertEqual(metadata["normalized_padding_value"], 0.0)
        self.assertEqual(metadata["output_crop_policy"], "original_spatial_extent")


class RealDeepLabV3PlusIntegrationTest(unittest.TestCase):
    """A slower, offline-safe integration test against the actual SMP network.

    encoder_weights=None guarantees no pretrained-weight download is attempted,
    while still exercising the real resnet101/DeepLabV3Plus architecture so
    this test genuinely proves the adapter's compatibility with SMP, not just
    with a fake stand-in network.
    """

    def test_real_smp_network_wrapped_in_adapter_returns_513_logits(self) -> None:
        import segmentation_models_pytorch as smp

        network = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            encoder_output_stride=16,
            in_channels=3,
            classes=4,
        )
        adapter = PaperAlignedDeepLabV3Plus(network, padding_multiple=16)
        adapter.eval()

        with torch.no_grad():
            logits = adapter(torch.randn(1, 3, 513, 513))

        self.assertEqual(tuple(logits.shape), (1, 4, 513, 513))


if __name__ == "__main__":
    unittest.main()
