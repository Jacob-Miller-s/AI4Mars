# Kaggle GPU Training Template

1. Enable a GPU accelerator and attach the existing AI4Mars dataset.
2. Inspect `/kaggle/input`, then set `AI4MARS_DATASET_ROOT` to the mounted dataset directory; do not use a hard-coded slug.
3. Clone the repository into `/kaggle/working/AI4Mars` and install only missing packages from `requirements.txt`.
4. Print `torch.cuda.is_available()`, GPU name, PyTorch version, and CUDA version.
5. Run an optional smoke test only when requested; otherwise launch the script baseline.

```bash
export AI4MARS_DATASET_ROOT=/kaggle/input/<your-dataset-mount>
cd /kaggle/working/AI4Mars
python -m src.train --config configs/kaggle_baseline.yaml --dataset-root "$AI4MARS_DATASET_ROOT" --output-root /kaggle/working/ai4mars
```

Save `/kaggle/working/ai4mars` as the notebook output or an output dataset. Attach that output dataset in a later session and pass its checkpoint with `--resume`.