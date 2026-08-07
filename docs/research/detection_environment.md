# Detection Environment Validation

## Sprint 1 Environment Contract

Use CPython 3.12 with the exact CUDA 12.1 binary pair in `requirements-detection.txt`:

- Python: 3.12
- torch: `2.5.1+cu121`
- torchvision: `0.20.1+cu121`
- CUDA runtime: 12.1

Create a dedicated environment, install this file once, then do not install the repository's unpinned `requirements.txt` afterward because it can replace the Torch/Torchvision binary pair.

```bash
py -3.12 -m venv .venv-detection
.venv-detection\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-detection.txt
python -m src.detection_smoke
```

The probe imports `torch` and `torchvision`, executes compiled `torchvision.ops.nms` and `torchvision.ops.roi_align`, constructs `maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)`, and performs a small CPU-safe inference. It prints Python, Torch, Torchvision, CUDA runtime, GPU name, operator status, construction status, and inference result. `--skip-forward` is available when only an operator/construction check is needed.

## Verified Local Probe

On this checkout, the probe ran successfully with:

- Python `3.14.6`
- torch `2.13.0+cpu`
- torchvision `0.28.0+cpu`
- CUDA runtime `None`; GPU unavailable
- NMS result `[0]`
- ROIAlign shape `[1, 2, 2, 2]`
- Mask R-CNN inference keys `boxes`, `labels`, `masks`, `scores`

This CPU environment is verified for development smoke testing but is not the selected Sprint 1 training environment. The selected Python 3.12/CUDA 12.1 environment must run the same probe on the intended GPU host before training. No custom NMS fallback is used or permitted.

## Kaggle Or GPU Host

After creating the environment or selecting Kaggle's preinstalled pair, run:

```bash
python -m src.detection_smoke
```

Record the full output with the future run artifacts. Do not start detector training if NMS, ROIAlign, construction, or forward status is not `ok`.