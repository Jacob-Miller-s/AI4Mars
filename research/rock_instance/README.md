# Rock-Instance Sprint 0

Sprint 0 tests the feasibility of an instance-level rock representation. It is data and protocol work only: it does not train a detector, tune the semantic baseline, or use sealed expert splits for development decisions.

The working hypothesis is that a physically defined discrete obstacle may be represented more appropriately as an object instance than as a monocular semantic terrain class. It is not a claim that Mask R-CNN or any other detector is automatically better.

## Development Boundary

All Sprint 0 commands accept only `train` and `val` manifests. The utilities reject expert split names. `expert_min1`, `expert_min2`, and `expert_min3` remain final held-out evidence.

## Tools

```bash
python -m src.rock_instance.stereo_inventory \
  --dataset-root <dataset-root> \
  --manifest-root artifacts/manifests \
  --train-manifest splits/msl_navcam_v1/train_nav.csv \
  --val-manifest splits/msl_navcam_v1/val_nav.csv \
  --output-dir outputs/sprint0/stereo_inventory

python -m src.rock_instance.component_audit \
  --dataset-root <dataset-root> \
  --manifest-root artifacts/manifests \
  --train-manifest splits/msl_navcam_v1/train_nav.csv \
  --val-manifest splits/msl_navcam_v1/val_nav.csv \
  --output-dir outputs/sprint0/component_audit

python -m src.rock_instance.pilot_selection \
  --component-images-csv outputs/sprint0/component_audit/big_rock_component_images.csv \
  --stereo-inventory-csv outputs/sprint0/stereo_inventory/stereo_inventory.csv \
  --dataset-root <dataset-root> \
  --output-dir artifacts/rock_instance/pilot_v2_source_compatible \
  --target-size 150 --seed 42 --max-per-sequence 1
```

The inventory reports explicit metadata only. AI4Mars `rng-30m` files are binary range-validity masks, not metric depth products. The component audit uses 8-connected semantic Big Rock regions as candidate review regions only; no CSV it writes is instance ground truth.

Pilot selection verifies every candidate image/mask pair against `--dataset-root` before ranking. It records unavailable sources, their missing-file reason, and the resolved extraction root in `rock_instance_source_compatibility.json`. A source-incompatible cohort must be regenerated into a new artifact directory; existing reviewed or initial-evidence artifacts are never edited to remove individual rows.

## Pilot Manifest

`rock_instance_pilot_candidates.csv` is a proposed manual-review queue. Every record has `annotation_status=candidate_unreviewed`; reviewers must apply `annotation_protocol.md` before it can become an instance dataset. Selection is deterministic, limits concentration to one image per acquisition sequence, targets 100-200 development images, and seeks isolated, multi-region, Bedrock-boundary, border, large, and tiny cases. Confirmed geometry cases are included when they exist; RGB-only/geometry-unresolved cases are retained for a controlled RGB pilot.

## Sprint 0.5 Review Workflow

The initial protocol is preserved as [annotation_protocol_v1.0-initial.md](annotation_protocol_v1.0-initial.md). Corrected calibration work must use [annotation_protocol_v2.0-calibration-resolved.md](annotation_protocol_v2.0-calibration-resolved.md); it is not frozen and does not authorize pilot expansion or training.

Create a separate corrected-calibration state before the next human review action. This copies source manifests and the v2 protocol, links the immutable initial snapshot by SHA-256, and leaves `calibration_initial_v1` untouched:

```bash
python -m src.rock_instance.review_tool --initialize-calibration-resolution \
  --candidate-manifest artifacts/rock_instance/pilot_v2_source_compatible/rock_instance_pilot_candidates.csv \
  --component-candidates-csv outputs/sprint0/component_audit/big_rock_component_candidates.csv \
  --calibration-manifest artifacts/rock_instance/pilot_v2_source_compatible/rock_instance_pilot_calibration_candidates.csv \
  --initial-snapshot artifacts/rock_instance/calibration_initial_v1/review_state_initial_v1.json \
  --protocol-path research/rock_instance/annotation_protocol_v2.0-calibration-resolved.md \
  --dataset-root <dataset-root> --output-dir artifacts/rock_instance/calibration_resolved_v2
```

The state enforces component-by-component completion. `uncertain` is terminal but makes the entire image ineligible for ordinary Mask R-CNN target conversion. Human-authored split/merge resolutions are accepted only through explicit resolution records linked to initial decision IDs; no source component is split or merged automatically.

The versioned candidate and reviewed artifacts are under `artifacts/rock_instance/`. They reference source paths instead of copying imagery:

```bash
# Deterministic 24-image protocol-calibration subset from the preserved 150-image pilot.
python -m src.rock_instance.calibration_selection \
  --pilot-candidates-csv outputs/sprint0/pilot_selection/rock_instance_pilot_candidates.csv \
  --output-dir artifacts/rock_instance/pilot_v0_candidates --target-size 24 --seed 42

# Copy source-referenced candidates and create empty, resumable pilot_v1 review state.
python -m src.rock_instance.review_tool --initialize \
  --candidate-manifest outputs/sprint0/pilot_selection/rock_instance_pilot_candidates.csv \
  --dataset-root <dataset-root> --output-dir artifacts/rock_instance/pilot_v1_reviewed

# Persist the deterministic 24-image calibration scope before showing any candidate.
# This preserves the 150-image state and prevents the default queue from entering the other 126 images.
python -m src.rock_instance.review_tool --activate-calibration-scope \
  --state-path artifacts/rock_instance/pilot_v1_reviewed/review_state.json \
  --calibration-manifest artifacts/rock_instance/pilot_v0_candidates/rock_instance_pilot_calibration_candidates.csv \
  --dataset-root <dataset-root>

# Render the next pending candidate with RGB, NAV terrain context, and component-ID overlays.
python -m src.rock_instance.review_tool \
  --state-path artifacts/rock_instance/pilot_v1_reviewed/review_state.json \
  --component-candidates-csv outputs/sprint0/component_audit/big_rock_component_candidates.csv \
  --dataset-root <dataset-root> --show
```

Use the interactive reviewer for calibration adjudication. It is the preferred workflow: select a cyan candidate box, choose its human disposition, add notes and flags, and click **Save decision**. For an `accepted` rock, select `accepted`, click **Draw polygon**, then click visible boundary points in the right panel before saving. Each save is atomic and leaves the image `in_progress`; click **Finish image** only after all its candidate decisions are recorded, which advances to the next calibration image. No shell command is needed for individual decisions.

```bash
python -m src.rock_instance.review_tool \
  --state-path artifacts/rock_instance/pilot_v1_reviewed/review_state.json \
  --component-candidates-csv outputs/sprint0/component_audit/big_rock_component_candidates.csv \
  --dataset-root <dataset-root> --interactive
```

If an incomplete image needs a clean re-review, archive its attempts before resetting only that image. This command refuses to reset completed images and never changes the immutable initial snapshot:

```bash
python -m src.rock_instance.review_tool \
  --state-path artifacts/rock_instance/calibration_resolved_v2/review_state.json \
  --dataset-root <dataset-root> \
  --restart-image --image-id <image-id> \
  --restart-reason "reason for clean re-review"
```

For each human decision, call the same reviewer command with an `--action`: `accepted`, `rejected_bedrock`, `rejected_noise`, `split_required`, `merge_required`, `uncertain`, or `deferred`. Supply either `--component-id` or a reviewer-defined `--bbox`; accepted rocks additionally require `--polygon-json` so their visible-object geometry is explicit. `--truncated`, `--occluded`, and `--notes` record qualifying context and rationale. Every action is atomically saved and can be resumed later. The reviewer never modifies source semantic masks.

```bash
python -m src.rock_instance.review_report \
  --state-path artifacts/rock_instance/pilot_v1_reviewed/review_state.json \
  --calibration-manifest artifacts/rock_instance/pilot_v0_candidates/rock_instance_pilot_calibration_candidates.csv \
  --output-json artifacts/rock_instance/pilot_v1_reviewed/review_progress.json
```

The report remains intentionally empty until a reviewer records decisions. Once calibration review begins, it reports actual decision counts, sequence coverage, visible-instance area statistics, and a clearly labelled preliminary extrapolation only when calibration images have been reviewed.

## Calibration Closure Gate

After all 24 primary calibration images have terminal component coverage, create the deterministic eight-image isolated intra-rater repeat state and write the pre-repeat closure report:

```bash
python -m src.rock_instance.calibration_closure \
  --primary-state-path artifacts/rock_instance/calibration_resolved_v2/review_state.json \
  --component-candidates-csv artifacts/rock_instance/calibration_resolved_v2/big_rock_component_candidates.csv \
  --prepare-repeat-output-dir artifacts/rock_instance/calibration_repeat_v1 \
  --repeat-target-size 8 \
  --closure-output-json artifacts/rock_instance/calibration_resolved_v2/calibration_closure_pre_repeat.json
```

The closure report records primary decision counts, unresolved-state status, repeat selection provenance, and the freeze gate. It remains blocked until the isolated repeat review is complete and its agreement analysis is documented. It does not authorize the remaining 126 pilot images, protocol freeze, target export, or any model training.

After the isolated repeat is complete, analyze intra-rater consistency without changing either annotation state, then bind the hash-validated result into the closure gate:

```bash
python -m src.rock_instance.intra_rater_consistency \
  --primary-state-path artifacts/rock_instance/calibration_resolved_v2/review_state.json \
  --repeat-state-path artifacts/rock_instance/calibration_repeat_v1/review_state.json \
  --repeat-selection-path artifacts/rock_instance/calibration_repeat_v1/repeat_selection.json \
  --output-dir artifacts/rock_instance/calibration_repeat_v1/intra_rater_consistency \
  --markdown-path docs/research/rock_instance/intra_rater_consistency.md

python -m src.rock_instance.calibration_closure \
  --primary-state-path artifacts/rock_instance/calibration_resolved_v2/review_state.json \
  --repeat-state-path artifacts/rock_instance/calibration_repeat_v1/review_state.json \
  --agreement-report-path artifacts/rock_instance/calibration_repeat_v1/intra_rater_consistency/intra_rater_consistency.json \
  --closure-output-json artifacts/rock_instance/calibration_resolved_v2/calibration_closure_post_repeat.json
```

The analysis compares direct component dispositions separately from accepted-instance counts and polygon mask IoU. It keeps split/merge and multi-annotation structures out of categorical agreement, records them as discrepancies, and never chooses either pass as correct. A `CLARIFY` recommendation keeps the gate blocked pending protocol language changes; a `FREEZE` recommendation still requires explicit human approval before any protocol freeze or pilot expansion.

## Future Comparison

mIoU and AP are not directly comparable. Once reviewed annotations exist, report object-centric rock recall, precision, missed-rock rate, false rock detections per image, localization/mask quality, errors overlapping semantic Bedrock, and performance by apparent size. A semantic-to-object connected-component adapter can only be used as a separately documented controlled comparison, with connectivity and filtering assumptions made explicit. Later geometry work should add physical width, height, distance, and uncertainty only after calibrated stereo/range products are verified.
