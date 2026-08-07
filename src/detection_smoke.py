"""Validate compiled torchvision detection operators without training a model."""

from __future__ import annotations

import argparse
import platform
from typing import Any


def run_detection_smoke(*, run_forward: bool = True) -> dict[str, Any]:
    """Exercise NMS, ROIAlign, Mask R-CNN construction, and optional CPU-safe inference."""
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.ops import nms, roi_align

    nms_result = nms(
        torch.tensor([[0.0, 0.0, 8.0, 8.0], [1.0, 1.0, 7.0, 7.0]]),
        torch.tensor([0.9, 0.8]),
        0.5,
    )
    roi_result = roi_align(
        torch.rand(1, 2, 8, 8),
        [torch.tensor([[0.0, 0.0, 6.0, 6.0]])],
        output_size=(2, 2),
    )
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=64, max_size=64)
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nms_status": "ok",
        "nms_kept_indices": nms_result.tolist(),
        "roi_align_status": "ok",
        "roi_align_shape": list(roi_result.shape),
        "maskrcnn_construction_status": "ok",
        "minimal_forward_status": "not_requested",
    }
    if run_forward:
        model.eval()
        with torch.inference_mode():
            output = model([torch.rand(3, 64, 64)])
        required_keys = {"boxes", "labels", "scores", "masks"}
        observed_keys = set(output[0]) if len(output) == 1 else set()
        if not required_keys.issubset(observed_keys):
            raise RuntimeError(f"Mask R-CNN inference omitted required keys: {sorted(required_keys - observed_keys)}")
        report["minimal_forward_status"] = "ok"
        report["minimal_forward_keys"] = sorted(observed_keys)
        report["minimal_forward_shapes"] = {name: list(value.shape) for name, value in output[0].items()}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-forward", action="store_true", help="Only validate operators and model construction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_detection_smoke(run_forward=not args.skip_forward)
    for name, value in report.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()