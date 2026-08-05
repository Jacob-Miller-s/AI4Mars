# AI4Mars Paper-Aligned Reproduction

## Scope And Claim

This work is a reproduction: an independently implemented experiment designed to align with the reported AI4Mars semantic baseline. It is not an exact replication because the publication does not specify all training details. An extension changes the question or method; the planned rock-instance system is an extension and is deliberately excluded.

The paper supports these claims: DeepLabv3+ with a ResNet-101 backbone initialized from ImageNet; 1024x1024 images resized to 513x513; MSL experiments and MSL NAVCAM baseline tables; merged crowdsourced training labels; expert test masks at min1/min2/min3 agreement; overall pixel accuracy, random-validation mIoU, and row-normalized confusion matrices; and class weights `weight[c] = 1 - class_composition[c]`.

## Implementation Choices

The paper omits optimizer, learning rate, schedule, weight decay, epochs, augmentation, exact batch size, seed, output stride, framework versions, early stopping, and checkpoint policy. This implementation records those values as configuration choices, not paper facts. Defaults are AdamW, 1e-4 learning rate, cosine schedule, 1e-4 weight decay, seed 42, output stride 16, no augmentation, 40 epochs, and validation-mIoU checkpoint selection. These choices may be changed only through a saved configuration.

For a P100, begin with the configured physical batch size of one, run a short validation or one-epoch smoke command, increase only after observing stable memory headroom, and set gradient accumulation explicitly when a larger effective batch is required. Record both physical batch size and accumulation steps; never silently change the effective batch size between runs.

The current U-Net baseline uses a different architecture, 256x256 inputs, inverse-frequency normalized weighting, and notebook-led execution. The reproduction uses SMP DeepLabV3Plus, ResNet-101/ImageNet, 513x513 inputs, ImageNet normalization, and the paper complement-composition weighting. Both weighting strategies remain separately selectable.

## Dataset And Evaluation Protocol

The reproduction scope is MSL Curiosity NAVCAM with NAV labels only. MSL Mastcam, MER, M2020, M2020_GEO, and unrelated subsets are rejected by manifest validation. Train and development validation require merged crowdsourced labels. Expert min1, min2, and min3 masks are never touched by `src.train`/`src.paper_train` -- they are configured only for leakage checking and provenance hashing during training. They are evaluated exclusively by the separate `src.paper_evaluate` entry point, which loads one frozen checkpoint and reports each expert split independently. This separation is intentional and architectural (`RunLogger`'s best-epoch tracking is gated to the crowdsourced validation split role and never fires for expert-evaluation runs), so expert masks can never influence checkpoint selection, early stopping, or any other training decision.

Manifests must use dataset-relative POSIX paths, source and acquisition identifiers, deterministic ordering, matching original geometry, valid NAV IDs, and fingerprints. Train/development/test source and sequence leakage is rejected. The committed scoped manifests currently report 12,945 train rows, 3,112 validation rows, and 322 rows for each expert threshold, each with 322 unique source images. This does not explain the paper's stated 526-record expert set and is a known comparability limitation requiring source-archive verification before numerical comparison.

Rows of confusion matrices are ground truth and columns are predictions. Their diagonal values are class recall/class accuracy, not IoU. The runner records weighted and unweighted validation loss, pixel accuracy, mIoU, per-class IoU/precision/recall/support/predicted counts, raw confusion matrices, and row-normalized confusion matrices. `src.paper_evaluate` reports the identical set of statistics per expert split, plus per-split CSV per-class tables and labeled PNG confusion-matrix figures, and states explicitly in its JSON output that the normalized-confusion-matrix diagonal is recall, not IoU.

## Exact Commands For All Six Workflows

Every command below assumes execution from the repository root with the paper-reproduction Python environment active (the project `.venv` locally, or Kaggle's own interpreter after installing `requirements-kaggle.txt`). Substitute `--dataset-root`/`--manifest-root`/`--output-root` as needed; on Kaggle, `--dataset-root` is the mounted input directory and `--output-root` must stay beneath `/kaggle/working/ai4mars-paper-reproduction`.

1. Metadata-only manifest audit (cheap, no file I/O -- checks manifest schema, scope, leakage, and ordering only):
   ```
   python -m src.train --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --dataset-root <DATASET_ROOT> --validate-only --validation-level metadata
   ```
2. Full manifest audit (expensive -- additionally opens every image/mask file to check geometry and label IDs):
   ```
   python -m src.train --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --dataset-root <DATASET_ROOT> --validate-only --validation-level full
   ```
3. Bounded GPU pipeline smoke test (one epoch, four samples per split, proves wiring only):
   ```
   python -m src.train --config configs/reproduction/paper_deeplabv3plus_kaggle_smoke.yaml --dataset-root <DATASET_ROOT> --output-root /kaggle/working/ai4mars-paper-reproduction/smoke
   ```
4. Short full-data calibration run (three epochs over the complete train/val split, sanity-checks the training curve before committing to the full run):
   ```
   python -m src.train --config configs/reproduction/paper_deeplabv3plus_kaggle_calibration.yaml --dataset-root <DATASET_ROOT> --output-root /kaggle/working/ai4mars-paper-reproduction/calibration
   ```
5. Fixed full reproduction run (40 epochs, batch size 1, no early stopping -- the run whose checkpoints are candidates for expert evaluation):
   ```
   python -m src.train --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --dataset-root <DATASET_ROOT> --output-root /kaggle/working/ai4mars-paper-reproduction
   ```
6. Separate final expert evaluation (loads a frozen checkpoint from workflow 5 and reports min1/min2/min3 splits independently; never run automatically by any of the workflows above):
   ```
   python -m src.paper_evaluate --config configs/reproduction/paper_deeplabv3plus_kaggle_p100.yaml --checkpoint /kaggle/working/ai4mars-paper-reproduction/runs/paper-deeplabv3plus-kaggle-p100/checkpoints/best_val_miou.pth --dataset-root <DATASET_ROOT> --splits expert_min1 expert_min2 expert_min3
   ```

Expected output locations (all beneath the run's `--output-root`, under `runs/<run_id>/`): `metadata.json` and `summary.json` (run-level), `config.json` (frozen configuration snapshot), `metrics.jsonl` (per-epoch/per-split metrics), `checkpoints/last.pth` and `checkpoints/best_val_miou.pth` (training runs only), and `artifacts/` (manifest audit JSON for training runs; per-split `*_per_class_metrics.csv`, `*_confusion_matrix_raw.csv`, `*_confusion_matrix_normalized.csv`, `*_confusion_matrix.png`, and the combined `expert_evaluation.json` for `src.paper_evaluate` runs).

## Reproducibility Checklist

- Record Git commit, complete configuration, environment, GPU, model metadata, normalization, class mapping, output stride, and manifest hashes.
- Persist raw class counts, proportions, ignore pixels, weights, and the training-manifest fingerprint.
- Preserve optimizer, scheduler, AMP scaler, global step, RNG state, and best validation metric in checkpoints.
- Select checkpoints on crowdsourced validation only; do not tune repeatedly on expert masks. Expert masks are scored exactly once per checkpoint, only via `src.paper_evaluate`, never by `src.train`/`src.paper_train`.
- Write generated artifacts only beneath the configured output root.

Comparability is threatened by the unresolved expert-count discrepancy, unreported original hyperparameters, dataset archive/version differences, preprocessing-library behavior, hardware scale, and any difference in split construction. A local smoke run proves wiring only, never paper performance.

The local smoke configuration (`paper_deeplabv3plus_local_smoke.yaml`) caps every split at two samples and runs on CPU only; it is intentionally not paper-comparable and exists only to test paths, preprocessing, logging, checkpointing, and evaluation serialization. Its Kaggle counterpart, `paper_deeplabv3plus_kaggle_smoke.yaml`, caps every split at four samples, runs one epoch on GPU, and proves the full CUDA pipeline (AMP, checkpointing, validation) without touching paper-comparable performance. `paper_deeplabv3plus_kaggle_calibration.yaml` is a third, distinct tier: it runs the complete train/validation data (no sample cap) for three epochs so the loss/mIoU trend can be sanity-checked before committing to the full 40-epoch run.

## Remote Workflows

For local VS Code plus Kaggle: edit and run CPU checks locally, push the approved branch, update the Kaggle notebook with the API or browser, version its `/kaggle/working` outputs, then import the resulting run directory into the local dashboard. On Kaggle, install dependencies from `requirements-kaggle.txt`, never from `requirements.txt` -- the Kaggle image already ships a GPU-matched CUDA build of `torch`/`torchvision`, and `requirements.txt` (used for local development) would reinstall or downgrade it. `requirements-kaggle.txt` deliberately omits `torch`, `torchvision`, and `torchaudio` for this reason. For Remote-SSH: provision an SSH-accessible Linux GPU host, clone the repository, create the environment, configure dataset/output roots, open the host in VS Code Remote-SSH, and run the same command. A remote Jupyter server may be reached through a secure SSH tunnel; no provider credentials belong in this repository.