# Mars Terrain Segmentation for Rover Navigation

A computer vision project using NASA's AI4Mars dataset to train a semantic segmentation model that classifies Martian terrain in rover images. The pipeline covers data preprocessing, PyTorch model training, IoU evaluation, visual error analysis, and lays the groundwork for uncertainty-aware navigation support.

---

## What is Semantic Segmentation?

Semantic segmentation is a computer vision task where every pixel in an image is assigned a class label. Unlike object detection (which draws bounding boxes), segmentation produces a pixel-level map — useful for understanding _where_ each terrain type is in a Mars rover image.

## About the AI4Mars Dataset

The [AI4Mars terrain-segmentation dataset](https://zenodo.org/records/15995036) is the correct per-pixel segmentation release for this project. It contains Mars rover images and crowd-sourced terrain masks for navigation-oriented classes such as **soil**, **bedrock**, **sand**, and **big_rock**.

The older Zenodo record [4033453](https://zenodo.org/record/4033453) is a different MSL image-classification dataset and should not be used here.

### Research Question

How effectively can pretrained encoder-decoder segmentation models classify navigational terrain classes in AI4Mars under local hardware constraints, and which terrain classes remain failure modes?

> **Note:** You must download the dataset manually. See setup instructions below.

---

## Repository Structure

```
mars-terrain-segmentation/
│
├── notebooks/
│   ├── 00_nasa_api_discovery.ipynb      # Discover dataset metadata via NASA/Zenodo APIs
│   ├── 01_dataset_inspection.ipynb      # Inspect extracted files and verify structure
│   ├── 02_dataset_viewer.ipynb          # Visualize image/mask pairs and overlays
│   ├── 03_baseline_training.ipynb       # Train a minimal segmentation model
│   └── 04_evaluation_error_analysis.ipynb  # Evaluate predictions and analyse errors
│
├── src/
│   ├── data_paths.py     # Path constants and directory helpers
│   ├── nasa_catalog.py   # NASA/Zenodo API functions
│   ├── dataset.py        # PyTorch Dataset class and file helpers
│   ├── transforms.py     # Image/mask preprocessing utilities
│   ├── visualize.py      # Matplotlib visualization helpers
│   ├── metrics.py        # Pixel accuracy and IoU metrics
│   └── train_utils.py    # Training loop, checkpointing, evaluation
│
├── data/
│   ├── raw/              # Raw downloads (git-ignored)
│   ├── processed/        # Processed data (git-ignored)
│   └── samples/          # Small samples for quick testing (git-ignored)
│
├── models/               # Saved checkpoints (git-ignored)
│
├── outputs/
│   ├── figures/          # Saved plots (committed)
│   ├── predictions/      # Model predictions (git-ignored)
│   └── logs/             # Training logs (git-ignored)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions (VS Code + Jupyter)

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

---

## AI4Mars Research Console

The local Research Console reads durable run records, manifest evidence, and
safe artifact references. It is observational: it does not launch, stop, or
modify training jobs. Training launch control remains intentionally deferred.

Build the frontend once after installing the Python requirements:

```powershell
cd web
npm install
npm run build
cd ..
```

Start the console on loopback only:

```powershell
.\.venv\Scripts\python.exe -m src.research_console --host 127.0.0.1 --port 8000
```

On macOS or Linux, use:

```bash
python -m src.research_console --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The console deliberately keeps the expert test
set locked during iterative training and excludes protocol-invalid, legacy,
and sealed-test records from its default best-run ranking.

Create a clearly labeled CPU-only smoke record for local UI verification:

```powershell
.\.venv\Scripts\python.exe -m src.research_console.demo --runs-root outputs/runs --run-id synthetic-smoke
```

Synthetic runs carry `demo`, `synthetic`, and `non-benchmark` tags. Their
metrics are not AI4Mars research results and must not be compared as a
benchmark.

Run the repository tests with the supported test runner:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd web
npm test
npm run build
```

`pytest` is not a project dependency. See [docs/dashboard.md](docs/dashboard.md)
for operation and instrumentation, [docs/run-schema.md](docs/run-schema.md)
for the portable record contract, and
[docs/adr/0001-research-console-architecture.md](docs/adr/0001-research-console-architecture.md)
for the architecture decision.
