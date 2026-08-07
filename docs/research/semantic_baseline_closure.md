# Semantic Baseline Closure

## Frozen Result

The controlled semantic-segmentation phase is closed. The frozen DeepLabV3+ checkpoint must be identified by SHA-256, not rewritten or retuned:

- expected artifact location: `runs/paper-deeplabv3plus-kaggle-p100/checkpoints/best_val_miou.pth` beneath the configured output root
- SHA-256: `90e74a9071d9bfb180d80ab2bb1927f1ea83a74d7e0601750873c2547a5ddaa3`
- development-selected epoch: 25
- selection metric: validation mIoU `0.8328454614546958`
- validation IoU: soil `0.935420`, bedrock `0.944884`, sand `0.906688`, big_rock `0.544389`

The multi-GB checkpoint is an external run artifact and is intentionally not committed. Its hash and expected relative location are the artifact contract.

## Configuration And Provenance

- repository: `mandevautospa/AI4Mars`
- dataset: AI4Mars merged 0.6, MSL Curiosity NAVCAM, NAV label scheme
- task: soil, bedrock, sand, big_rock with ignore index 255
- configuration: `configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml`
- scoped manifests: `artifacts/manifests/splits/msl_navcam_v1/train_nav.csv` and `val_nav.csv`; sealed expert manifests are retained solely for final evaluation
- model: DeepLabV3+, ResNet-101 ImageNet initialization, input 513 x 513, output stride 16
- optimizer/schedule: AdamW, LR `1e-4`, weight decay `1e-4`, cosine schedule
- training: batch size 2, AMP, seed 42, 40 completed epochs

The exact source revision for a recovered artifact must be read from its checkpoint/run metadata. It is not inferred from the checkout used to write this closure record.

## Final Sealed Expert Evaluation

The checkpoint was frozen before expert evaluation. `expert_min1`, `expert_min2`, and `expert_min3` were used only for final reporting, never checkpoint selection, component thresholds, pilot selection, architecture choice, or subsequent tuning.

| Split | mIoU | Big Rock IoU | Big Rock Recall |
| --- | ---: | ---: | ---: |
| expert_min1 | 0.676079 | 0.111871 | 0.500998 |
| expert_min2 | 0.726952 | 0.113898 | 0.821594 |
| expert_min3 | 0.833381 | 0.418272 | 0.980598 |

| Split | GT Bedrock -> predicted Big Rock | GT Big Rock -> predicted Bedrock |
| --- | ---: | ---: |
| expert_min1 | 2.0048%, 301,448 pixels | 24.1374%, 23,107 pixels |
| expert_min2 | 1.7448%, 195,489 pixels | 7.2617%, 2,362 pixels |
| expert_min3 | 0.3591%, 23,563 pixels | 0%, 0 pixels |

## Interpretation And Boundary

The semantic baseline is strong for continuous terrain classes. Big Rock remains the outlier: recall can be high while IoU remains low because predicted Big Rock pixels include extensive false positives, predominantly from Bedrock. This motivates, but does not establish, the hypothesis that a physically discrete obstacle may be better represented as a reviewed object instance.

`python -m src.paper_error_analysis --evaluation-artifact <expert_evaluation.json> --output-dir <output-dir>` regenerates raw and row-normalized confusion CSVs, 300-dpi figures, and Bedrock/Big Rock summaries from an existing evaluation artifact only. It never loads a model or reruns inference.

The next research phase is Sprint 0 feasibility work in `research/rock_instance/`; no detector was trained in this closure.
