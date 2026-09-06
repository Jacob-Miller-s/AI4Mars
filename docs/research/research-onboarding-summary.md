# Research Onboarding Summary

This is the concise reference summary of the agreed AI4Mars research protocol. The complete numbered record is in [research-onboarding-questions-and-answers.md](research-onboarding-questions-and-answers.md).

## Research Direction

- Primary objective: improve big-rock IoU.
- First study: controlled four-class semantic segmentation.
- Analyze errors before changing the model.
- Change one meaningful variable per experiment.
- Record exact configuration, source revision, seed, data and manifest identities, environment, metrics, checkpoint hash, confusion matrix, and hypothesis/result notes.

## Data Boundaries

- Train split: parameter fitting.
- Crowdsourced validation: model selection, tuning, and first failure analysis.
- Expert min1/min2/min3: sealed final evaluation only.
- Expert results must not influence training, tuning, checkpoint selection, or candidate revision.

## Baseline and Artifacts

- Epoch-25 checkpoint: immutable historical reference evidence.
- Experimental control: best validation-mIoU checkpoint from the completed 40-epoch-budget run.
- Do not assume the selected checkpoint is epoch 40; record its actual epoch.
- Kaggle checkpoint path:

```text
/kaggle/working/ai4mars-paper-reproduction/runs/paper-deeplabv3plus-kaggle-p100/checkpoints/best_val_miou.pth
```

- Save the completed notebook output as an explicit Kaggle version and record its identity.
- Record the absolute run root, Kaggle dataset/output identity, source commit, exact configuration or hash, global step, checkpoint filename, validation metrics, and SHA-256.

## Validation Analysis

Write validation reports and confusion summaries under `expert-evaluation`, with validation and expert results separated into distinct subdirectories. Save top-10 overlays under `figures`.

Required outputs:

- per-class IoU and recall;
- raw and row-normalized confusion matrices;
- bedrock-to-big-rock and big-rock-to-bedrock counts and rates;
- per-image and per-sequence metrics;
- top 10 missed-rock, false-positive, boundary-confusion, fragmentation, and sequence-failure examples;
- checkpoint, manifest, and ranking metadata;
- separate observed evidence, interpretation, and recommendation records.

A failure category is dominant if it represents at least 50% of categorized big-rock error pixels or has at least twice the error mass of the next-largest category. Record affected pixel and image counts. Otherwise classify the evidence as mixed/inconclusive and do not tune yet.

Use deterministic rankings and image ID as the tie-breaker. Flag and audit questionable labels rather than silently changing them; approved corrections become a new versioned annotation set.

## First Intervention

Select exactly one intervention from the validation evidence:

- missed rocks: class weighting or loss rebalancing;
- sparse examples: targeted sampling or oversampling;
- boundary confusion: label audit;
- sequence-specific failures: domain-shift investigation;
- false positives: inspect confusing terrain and rock-like cues;
- mixed evidence: improve analysis before tuning.

Screen candidates against the 40-epoch-budget control:

- big-rock IoU: at least `+0.03`;
- overall mIoU: no worse than `-0.01`;
- each non-big-rock IoU: no worse than `-0.02`;
- pixel accuracy: no worse than `-0.01`.

These are screening thresholds, not proof of generalization. Repeat promising changes with predetermined seeds such as 7, 42, 123, and 2026.

## Future Model Design

The future system separates terrain and obstacle perception:

- Three-class terrain model: soil, sand, and bedrock.
- Original big-rock pixels: provisionally `ignore`, because underlying terrain is occluded and cannot generally be inferred.
- Preserve original four-class masks unchanged.
- Independent binary big-rock model: full-image semantic segmentation using original big-rock pixels as positive, soil/sand/bedrock as negative, and `255` as ignored.
- Primary binary metric: big-rock IoU; also report precision, recall, confusion counts, and per-image/per-sequence performance.

Do not begin Mask R-CNN or other instance segmentation until annotations identify individual rocks and define object semantics, touching/overlap/occlusion/truncation policy, quality control, splits, and instance metrics. Stereo-derived height, width, range, and position are future work pending calibrated stereo data and a validated reconstruction pipeline.

## Execution Order

1. Finish the 40-epoch-budget run.
2. Save the notebook output as an explicit Kaggle version.
3. Select and hash its best validation-mIoU checkpoint.
4. Preserve epoch 25 as historical evidence.
5. Analyze crowdsourced validation only.
6. Choose one intervention using the predeclared decision rule.
7. Train and screen one four-class candidate.
8. Lock the candidate checkpoint, configuration, metadata, and hash.
9. Evaluate locked control and candidate on sealed expert splits.
10. Begin the independent three-class terrain and full-image binary big-rock studies.
11. Defer instance segmentation and stereo geometry until the required data and protocols exist.
