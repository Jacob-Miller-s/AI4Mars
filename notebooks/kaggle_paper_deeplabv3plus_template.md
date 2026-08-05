# Kaggle Paper Reproduction Template

Attach the existing Zenodo AI4Mars mirror as a Kaggle dataset and set `DATASET_SLUG` to its mounted directory name. Kaggle inputs remain read-only; all generated output is under `/kaggle/working`.

```python
import os
import shutil
import subprocess
import torch

DATASET_SLUG = "REPLACE_WITH_AI4MARS_DATASET_SLUG"
REPOSITORY_URL = "https://github.com/mandevautospa/AI4Mars.git"
BRANCH = "feat/paper-deeplab-reproduction"
WORKDIR = "/kaggle/working/AI4Mars"
DATASET_ROOT = f"/kaggle/input/{DATASET_SLUG}"

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
print("CUDA:", torch.version.cuda, "PyTorch:", torch.__version__)
subprocess.run(["nvidia-smi"], check=False)
subprocess.run(["df", "-h", "/kaggle/input", "/kaggle/working"], check=False)
subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPOSITORY_URL, WORKDIR], check=True)
# requirements-kaggle.txt intentionally excludes torch/torchvision/torchaudio so this
# install never reinstalls or downgrades Kaggle's GPU-matched CUDA build. Never point
# this at requirements.txt on Kaggle -- that file pins torch/torchvision for local dev.
subprocess.run(["python", "-m", "pip", "install", "-r", f"{WORKDIR}/requirements-kaggle.txt"], check=True)
subprocess.run(["python", "-m", "src.train", "--config", "configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml", "--dataset-root", DATASET_ROOT, "--output-root", "/kaggle/working/ai4mars-paper-reproduction"], cwd=WORKDIR, check=True)
```

Use the notebook's Save Version action to preserve `/kaggle/working/ai4mars-paper-reproduction` as an output dataset. To resume, attach that output dataset, pass its checkpoint with `--resume-checkpoint`, and use a new run ID/output root. The template intentionally delegates all training logic to `src.train` and contains no credentials or local Windows paths.

Other reproduction workflows swap only the `--config` path and, for expert evaluation, the entry point:

```python
# Metadata-only manifest audit (cheap, no file I/O)
subprocess.run(["python", "-m", "src.train", "--config", "configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml", "--dataset-root", DATASET_ROOT, "--validate-only", "--validation-level", "metadata"], cwd=WORKDIR, check=True)

# Full manifest audit (expensive, opens every image/mask file)
subprocess.run(["python", "-m", "src.train", "--config", "configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml", "--dataset-root", DATASET_ROOT, "--validate-only", "--validation-level", "full"], cwd=WORKDIR, check=True)

# Bounded GPU pipeline smoke test
subprocess.run(["python", "-m", "src.train", "--config", "configs/reproduction/paper_deeplabv3plus_kaggle_smoke.yaml", "--dataset-root", DATASET_ROOT, "--output-root", "/kaggle/working/ai4mars-paper-reproduction/smoke"], cwd=WORKDIR, check=True)

# Short full-data calibration run
subprocess.run(["python", "-m", "src.train", "--config", "configs/reproduction/paper_deeplabv3plus_kaggle_calibration.yaml", "--dataset-root", DATASET_ROOT, "--output-root", "/kaggle/working/ai4mars-paper-reproduction/calibration"], cwd=WORKDIR, check=True)

# Sealed expert evaluation of a frozen checkpoint from the full run above (never run
# automatically as part of training/calibration/smoke testing)
subprocess.run(["python", "-m", "src.paper_evaluate", "--config", "configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml", "--checkpoint", "/kaggle/working/ai4mars-paper-reproduction/runs/paper-deeplabv3plus-kaggle-p100/checkpoints/best_val_miou.pth", "--dataset-root", DATASET_ROOT, "--splits", "expert_min1", "expert_min2", "expert_min3"], cwd=WORKDIR, check=True)
```