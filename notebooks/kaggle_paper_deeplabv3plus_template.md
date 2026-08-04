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
subprocess.run(["python", "-m", "pip", "install", "-r", f"{WORKDIR}/requirements.txt"], check=True)
subprocess.run(["python", "-m", "src.train", "--config", "configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml", "--dataset-root", DATASET_ROOT, "--output-root", "/kaggle/working/ai4mars-paper-reproduction"], cwd=WORKDIR, check=True)
```

Use the notebook's Save Version action to preserve `/kaggle/working/ai4mars-paper-reproduction` as an output dataset. To resume, attach that output dataset, pass its checkpoint with `--resume-checkpoint`, and use a new run ID/output root. The template intentionally delegates all training logic to `src.train` and contains no credentials or local Windows paths.