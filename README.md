# AI4Mars Rock Perception

Martian terrain segmentation and instance-level rock detection from rover imagery.

## Introduction

The objective of this work is to develop an instance-level perception method for identifying discrete
rover-scale rocks in Martian surface imagery. A semantic segmentation baseline is first established
using the AI4Mars dataset as a necessary reference for understanding the limitations of pixel-wise
terrain classification. In particular, the baseline is used to measure the degree to which Big Rock
is confused with Bedrock and other terrain classes. These observations guide the subsequent
development of a separate rock instance classifier intended to identify individual rock objects,
preserve their boundaries, and eventually incorporate stereo-derived estimates of physical size.
## Dataset

Experiments use the merged AI4Mars 0.6 release and are currently restricted to the Mars Science Laboratory NAVCAM subset.
The dataset is compromised of ~35K images from NASA's Planetary Data System (PDS), including grey-scale navigation camera
(NAVCAM) and color mast camera (Mastcam) from the Curiosity, Opportunity, and Spirit Mars rovers.

| Value | Class |
|---:|---|
| 0 | Soil |
| 1 | Bedrock |
| 2 | Sand |
| 3 | Big Rock |
| 255 | Ignore |

Dataset files are not included in this repository.


## Semantic Baseline

| Component | Configuration |
|---|---|
| Model | DeepLabV3+ |
| Encoder | ResNet-101 |
| Initialization | ImageNet |
| Input | `513 × 513` |
| Output stride | 16 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Batch size | 2 |
| Schedule | Cosine annealing |
| Precision | Automatic mixed precision |
| Epochs | 40 |
| Seed | 42 |

Inputs are padded internally to satisfy encoder stride requirements and cropped back to `513 × 513` before loss and metric computation.

## Preliminary Results

A three-epoch development run reached a mean IoU of `0.7837`.

| Class | IoU |
|---|---:|
| Soil | 0.9256 |
| Bedrock | 0.9359 |
| Sand | 0.8822 |
| Big Rock | 0.3911 |

Big Rock remains the principal failure mode and is predominantly confused with Bedrock. These are development-validation results, not final expert-test results.

### Reference

R. M. Swan et al., “AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars,” CVPR Workshops, 2021.

### 1. Clone the Repository

```bash
git clone https://github.com/mandevautospa/AI4Mars.git
cd AI4Mars
```

### 2. Create and Activate a Virtual Environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If you get a script execution error, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register the Kernel with Jupyter

```powershell
python -m ipykernel install --user --name mars-seg --display-name "Python (mars-seg)"
```

### 5. Open in VS Code

Open the repository folder in VS Code, then open any notebook. Select the **Python (mars-seg)** kernel when prompted.

---

## Notebook Workflow

| # | Notebook | Goal |
|---|----------|------|
| 00 | `00_nasa_api_discovery.ipynb` | Use NASA CKAN and Zenodo APIs to locate the correct terrain-segmentation record and download links |
| 01 | `01_dataset_inspection.ipynb` | Inspect extracted files, count images/masks, verify pairing logic |
| 02 | `02_dataset_viewer.ipynb` | Visualise NAV and M2020_GEO pairs, overlays, and class distributions |
| 03 | `03_baseline_training.ipynb` | Train a baseline pretrained U-Net and record reproducible settings |
| 04 | `04_evaluation_error_analysis.ipynb` | Evaluate predictions with pixel accuracy, per-class IoU, and error analysis |

---

## Current Milestone

> **Load Mars rover image/mask pairs, visualise overlays, verify class labels, and benchmark pretrained segmentation models under local hardware constraints.**

Start with `00_nasa_api_discovery.ipynb` to find the correct Zenodo record and download links, then manually download and extract the AI4Mars dataset into `data/raw/`. Then run `01_dataset_inspection.ipynb` to verify the structure before proceeding.

---

## Kaggle GPU Training

Attach the existing AI4Mars Kaggle dataset as an input. Kaggle mounts inputs read-only under `/kaggle/input/<dataset-slug>/`; set `AI4MARS_DATASET_ROOT` to either that mount or directly to its extracted `ai4mars-dataset-merged-0.6` directory. Do not guess the slug: inspect `/kaggle/input` in the notebook first.

Clone this repository into `/kaggle/working/AI4Mars` and run:

```bash
python -m src.train --config configs/kaggle_baseline.yaml \
	--dataset-root "$AI4MARS_DATASET_ROOT" \
	--output-root /kaggle/working/ai4mars \
	--run-id msl_navcam_kaggle_baseline
```

Outputs are written under `/kaggle/working/ai4mars/runs/<run-id>/`, including `metadata.json`, sampled `metrics.jsonl`, `system_metrics.jsonl`, `summary.json`, and `checkpoints/`. Save notebook outputs as a Kaggle version or create an output dataset to preserve them between sessions. Resume from a copied checkpoint with the same experiment-defining configuration and manifests:

```bash
python -m src.train --config configs/kaggle_baseline.yaml \
	--dataset-root "$AI4MARS_DATASET_ROOT" \
	--output-root /kaggle/working/ai4mars \
	--run-id msl_navcam_kaggle_baseline_resume \
	--resume /kaggle/input/<previous-output-dataset>/best.pth
```

The baseline retains the committed MSL NavCam NAV manifests, $256 \times 256$ resolution, U-Net/resnet34 ImageNet encoder, Adam at $10^{-3}$, `ignore_index=255`, and the configured class-weight vector. `configs/kaggle_384.yaml` is an explicit non-baseline resolution experiment. Start with batch size 4; after a CUDA OOM, lower only `batch_size` and record that change. GPU is supported first because TPU would require `torch-xla`, process spawning, XLA data loading and metric synchronization, device-specific checkpoints, and performance validation.

---

## Future Work

- **Per-class IoU reporting** — detailed breakdown by terrain class and failure mode
- **Uncertainty-aware segmentation** — predict confidence alongside class labels for safer navigation
- **Rover-to-rover generalisation** — transfer between Curiosity, Opportunity, and Spirit data
- **Hazard / traversability maps** — convert class predictions into actionable navigation masks
- **Cleaner experiment series** — compare pretrained U-Net, EfficientNet encoders, DeepLabV3+, and Dice/Focal/CE hybrids

---

## AI4Mars Senior Research Agent

The repository includes a persistent, read-only research advisor in `research_agent/`. It can inspect project files and notebooks, check segmentation metrics, estimate tensor memory, search current literature, critique experimental reasoning, and explain the mathematics behind recommendations.

After installing the requirements, start an ongoing conversation with:

```powershell
python -m research_agent
```

Ask one question non-interactively with:

```powershell
python -m research_agent --ask "What is the highest-value experiment to run next?"
```

Conversation history is stored locally under `.research_agent/` and is not committed.
