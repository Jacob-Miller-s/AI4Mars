# AI4Mars Semantic Reproduction

This repository reproduces the four-class MSL Curiosity NavCam semantic-segmentation result using DeepLabV3+ with a ResNet-101 encoder. It has one active research path: validate the frozen result locally, reproduce training on Kaggle, and evaluate the frozen checkpoint separately against sealed expert masks.

## Baseline Evidence

- Dataset: AI4Mars merged 0.6, DOI `10.5281/zenodo.15995036`, CC BY 4.0
- Scope: MSL Curiosity NavCam with NAV labels
- Classes: soil, bedrock, sand, big rock; ignore index `255`
- Model: DeepLabV3+, ResNet-101, ImageNet initialization, 513 x 513 input, output stride 16
- Frozen checkpoint: epoch 25
- Checkpoint SHA-256: `90e74a9071d9bfb180d80ab2bb1927f1ea83a74d7e0601750873c2547a5ddaa3`
- Development validation mIoU: `0.8328454614546958`

The checkpoint is an external artifact and is not stored in Git. Until the prepared `semantic-reproduction-v1` release is published, place it at `artifacts/ai4mars-paper-reproduction/frozen/deeplabv3plus-tesla-p100-seed42-best-val-miou.pth`. The onboarding workflow verifies its hash before loading it; after publication, it can also acquire the same file from the release URL.

## First-Hour Onboarding

Python 3.11 or newer is required. Full training is not part of onboarding and does not require a local GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Open [notebooks/01_onboarding.ipynb](notebooks/01_onboarding.ipynb), select the `.venv` kernel, and run all cells. It will:

1. Acquire and verify the epoch-25 checkpoint.
2. Verify all hashes in the fixed eight-example Onboarding Sample.
3. Run inference on CPU.
4. Check output shapes, finite values, class IDs, and hardware-tolerant metric ranges.
5. Display image, ground truth, prediction, and per-example mIoU.
6. Print an explicit `PASS`.

A verified local run produced pixel accuracy `0.8751` and mIoU `0.7507`. The Onboarding Sample is for setup validation and visual inspection only; it is not evaluation evidence.

## Canonical Notebooks

| Notebook | Environment | Purpose |
| --- | --- | --- |
| [01_onboarding.ipynb](notebooks/01_onboarding.ipynb) | Local CPU | Validate setup, checkpoint identity, fixed samples, inference, and predictions |
| [02_full_reproduction.ipynb](notebooks/02_full_reproduction.ipynb) | Kaggle GPU | Validate manifests, train or resume, preserve checkpoints, and plot completed epochs |
| [03_sealed_expert_evaluation.ipynb](notebooks/03_sealed_expert_evaluation.ipynb) | Kaggle GPU | Evaluate one already-frozen checkpoint on expert min1/min2/min3 splits |

The notebooks are thin adapters. The installed `ai4mars` package owns the workflow, and tests exercise the same interface.

## Full Reproduction

Attach the extracted AI4Mars merged 0.6 dataset to Kaggle and set `DATASET_ROOT` in the notebook. Generated files must stay under `/kaggle/working`; `/kaggle/input` is read-only.

The canonical configuration is [configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml](configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml). It records 40 epochs, physical batch size 2, AdamW at `1e-4`, cosine scheduling, AMP, seed 42, and validation-mIoU checkpoint selection. Resume preserves optimizer, scheduler, AMP scaler, global step, and random-number-generator state.

Training writes minimal scientific records under `runs/<run-id>/`: frozen configuration, provenance and manifest hashes, completed-epoch metrics, `last.pth`, `best_val_miou.pth`, and a final summary. It does not collect live system telemetry or maintain a dashboard.

Sealed expert evaluation is a separate invocation and never participates in checkpoint selection or tuning. See [docs/research/ai4mars_paper_reproduction.md](docs/research/ai4mars_paper_reproduction.md) for the protocol and command-line equivalents.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The regular suite uses lightweight fixtures. The real 549 MB checkpoint is exercised for release validation and by the onboarding notebook rather than downloaded on every test run.

## Repository Map

```text
ai4mars/                 Installed implementation and notebook-facing interface
artifacts/manifests/     Fixed dataset and split manifests
configs/reproduction/    Reproduction, calibration, and smoke configurations
data/samples/onboarding/ Eight attributed, hash-verified development examples
docs/research/           Scientific protocol and frozen-result record
notebooks/               The three active researcher workflows
tests/                   Interface, training, evaluation, and provenance tests
```

Historical discovery notebooks, U-Net baselines, dashboard experiments, the research agent, and rock-instance feasibility work are preserved in Git history at tag `pre-semantic-reproduction-mvp-2026-09-05`, not in the active workflow.
