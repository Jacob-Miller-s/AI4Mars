# AI4Mars Paper-Aligned Reproduction

## Scope And Claim

This work is a reproduction: an independently implemented experiment designed to align with the reported AI4Mars semantic baseline. It is not an exact replication because the publication does not specify all training details. An extension changes the question or method; the planned rock-instance system is an extension and is deliberately excluded.

The paper supports these claims: DeepLabv3+ with a ResNet-101 backbone initialized from ImageNet; 1024x1024 images resized to 513x513; MSL experiments and MSL NAVCAM baseline tables; merged crowdsourced training labels; expert test masks at min1/min2/min3 agreement; overall pixel accuracy, random-validation mIoU, and row-normalized confusion matrices; and class weights `weight[c] = 1 - class_composition[c]`.

## Implementation Choices

The paper omits optimizer, learning rate, schedule, weight decay, epochs, augmentation, exact batch size, seed, output stride, framework versions, early stopping, and checkpoint policy. This implementation records those values as configuration choices, not paper facts. Defaults are AdamW, 1e-4 learning rate, cosine schedule, 1e-4 weight decay, seed 42, output stride 16, no augmentation, 40 epochs, and validation-mIoU checkpoint selection. These choices may be changed only through a saved configuration.

For a P100, begin with the configured physical batch size of one, run a short validation or one-epoch smoke command, increase only after observing stable memory headroom, and set gradient accumulation explicitly when a larger effective batch is required. Record both physical batch size and accumulation steps; never silently change the effective batch size between runs.

The current U-Net baseline uses a different architecture, 256x256 inputs, inverse-frequency normalized weighting, and notebook-led execution. The reproduction uses SMP DeepLabV3Plus, ResNet-101/ImageNet, 513x513 inputs, ImageNet normalization, and the paper complement-composition weighting. Both weighting strategies remain separately selectable.

## Dataset And Evaluation Protocol

The reproduction scope is MSL Curiosity NAVCAM with NAV labels only. MSL Mastcam, MER, M2020, M2020_GEO, and unrelated subsets are rejected by manifest validation. Train and development validation require merged crowdsourced labels. Expert min1, min2, and min3 masks are separate final evaluations; the validation-selected model is evaluated once on each and those results are labeled by agreement set.

Manifests must use dataset-relative POSIX paths, source and acquisition identifiers, deterministic ordering, matching original geometry, valid NAV IDs, and fingerprints. Train/development/test source and sequence leakage is rejected. The committed scoped manifests currently report 12,945 train rows, 3,112 validation rows, and 322 rows for each expert threshold, each with 322 unique source images. This does not explain the paper's stated 526-record expert set and is a known comparability limitation requiring source-archive verification before numerical comparison.

Rows of confusion matrices are ground truth and columns are predictions. Their diagonal values are class recall/class accuracy, not IoU. The runner records weighted and unweighted validation loss, pixel accuracy, mIoU, per-class IoU/precision/recall/support/predicted counts, raw confusion matrices, and row-normalized confusion matrices.

## Reproducibility Checklist

- Record Git commit, complete configuration, environment, GPU, model metadata, normalization, class mapping, output stride, and manifest hashes.
- Persist raw class counts, proportions, ignore pixels, weights, and the training-manifest fingerprint.
- Preserve optimizer, scheduler, AMP scaler, global step, RNG state, and best validation metric in checkpoints.
- Select checkpoints on crowdsourced validation only; do not tune repeatedly on expert masks.
- Write generated artifacts only beneath the configured output root.

Comparability is threatened by the unresolved expert-count discrepancy, unreported original hyperparameters, dataset archive/version differences, preprocessing-library behavior, hardware scale, and any difference in split construction. A local smoke run proves wiring only, never paper performance.

The local smoke configuration caps every split at two samples. It is intentionally not paper-comparable and exists only to test paths, preprocessing, logging, checkpointing, and evaluation serialization on CPU.

## Remote Workflows

For local VS Code plus Kaggle: edit and run CPU checks locally, push the approved branch, update the Kaggle notebook with the API or browser, version its `/kaggle/working` outputs, then import the resulting run directory into the local dashboard. For Remote-SSH: provision an SSH-accessible Linux GPU host, clone the repository, create the environment, configure dataset/output roots, open the host in VS Code Remote-SSH, and run the same command. A remote Jupyter server may be reached through a secure SSH tunnel; no provider credentials belong in this repository.