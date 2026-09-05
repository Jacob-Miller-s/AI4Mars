# AI4Mars Paper-Aligned Reproduction

## Scope

This is an independently implemented reproduction of the reported AI4Mars semantic baseline, not an exact replication. The publication establishes DeepLabV3+ with a ResNet-101 ImageNet encoder, 513 x 513 inputs, MSL NavCam terrain labels, complement-composition class weighting, pixel accuracy, mIoU, and confusion matrices. Optimizer, schedule, seed, batch size, augmentation, output stride, epoch count, and checkpoint policy are recorded implementation choices where the paper is silent.

The active scope is MSL Curiosity NavCam with NAV labels only: soil, bedrock, sand, and big rock, with ignore index 255. Training and development validation use merged crowdsourced labels. Expert min1, min2, and min3 masks are sealed final evidence.

## Model Contract

The canonical configuration uses DeepLabV3+, ResNet-101, ImageNet initialization, output stride 16, and 513 x 513 inputs. `segmentation_models_pytorch` requires dimensions divisible by the output stride, so `PaperAlignedDeepLabV3Plus` pads normalized inputs on the right and bottom from 513 to 528, forwards through the unchanged network, and crops logits back to 513 x 513. Masks, loss, confusion matrices, and metrics never include the padded border.

Canonical training uses physical batch size 2 because train-mode BatchNorm in the ASPP pooled branch cannot train with tensors shaped `[1, C, 1, 1]`. Gradient accumulation changes effective optimizer batch size but does not solve that per-forward BatchNorm constraint. Validation and expert evaluation may use batch size 1 in eval mode.

## Protocol Boundary

Training may read only the train and crowdsourced validation splits. Expert manifests are included in leakage and provenance checks but their pixels are scored only by the separate expert-evaluation workflow. A frozen checkpoint cannot be selected, stopped, or tuned using expert results.

Rows of every confusion matrix are ground-truth classes and columns are predictions. The diagonal of a row-normalized confusion matrix is recall, not IoU. Per-class IoU is reported separately.

## Workflows

The three canonical notebooks are the researcher-facing adapters:

1. `notebooks/01_onboarding.ipynb` validates the installed package, frozen checkpoint, fixed development samples, CPU inference, and prediction inspection.
2. `notebooks/02_full_reproduction.ipynb` performs manifest validation, full Kaggle training or resume, checkpoint creation, and static plots from completed epoch records.
3. `notebooks/03_sealed_expert_evaluation.ipynb` evaluates one frozen checkpoint against expert min1/min2/min3 and writes final artifacts.

Equivalent command-line entry points are:

```text
python -m ai4mars.paper_train --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --dataset-root <DATASET_ROOT> --validate-only --validation-level metadata
python -m ai4mars.paper_train --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --dataset-root <DATASET_ROOT> --validate-only --validation-level full
python -m ai4mars.paper_train --config configs/reproduction/paper_deeplabv3plus_kaggle_smoke.yaml --dataset-root <DATASET_ROOT> --output-root <OUTPUT_ROOT>
python -m ai4mars.paper_train --config configs/reproduction/paper_deeplabv3plus_kaggle_calibration.yaml --dataset-root <DATASET_ROOT> --output-root <OUTPUT_ROOT>
python -m ai4mars.paper_train --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --dataset-root <DATASET_ROOT> --output-root <OUTPUT_ROOT>
python -m ai4mars.paper_evaluate --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --checkpoint <FROZEN_CHECKPOINT> --dataset-root <DATASET_ROOT> --splits expert_min1 expert_min2 expert_min3
```

## Scientific Records

Training writes `metadata.json`, `config.json`, `metrics.jsonl`, `summary.json`, manifest audits, and `checkpoints/last.pth` plus `checkpoints/best_val_miou.pth`. Resume restores optimizer, scheduler, AMP scaler, global step, best validation metric, and RNG state.

Expert evaluation writes one report and per-split class metrics and confusion artifacts. Existing evaluation JSON can be re-rendered without inference:

```text
python -m ai4mars.paper_error_analysis --evaluation-artifact <expert_evaluation.json> --output-dir <OUTPUT_DIR>
```

The frozen result and expert numbers are recorded in [semantic_baseline_closure.md](semantic_baseline_closure.md). The release evidence bundle pairs that record with the exact checkpoint SHA-256, frozen configuration, and checkpoint provenance.

## Reproducibility Rules

- Use the committed manifests without row reordering or split substitution.
- Keep 513 x 513 as the reported scientific resolution; 528 x 528 is internal padding only.
- Select checkpoints on crowdsourced validation mIoU only.
- Evaluate expert masks only after the checkpoint is frozen.
- Treat the fixed Onboarding Sample as setup evidence, never as a benchmark.
- Preserve configuration, manifest hashes, source revision, environment, epoch metrics, and final summaries with every run.
