# Research Onboarding Questions and Answers

Complete record of the grilling session that defined the AI4Mars research protocol.

## Questions 1-15: Research Direction

**Q1. What is your first research question?**

Improve big-rock IoU. The first implementation remains a four-class semantic-segmentation study.

**Q2. What counts as improvement?**

Higher big-rock IoU is the primary objective. Also report overall mIoU, per-class IoU, pixel accuracy, precision, recall, and confusion matrices. Do not accept a big-rock gain if it causes unacceptable regressions elsewhere.

**Q3. What is your baseline boundary?**

The frozen checkpoint is the control. Keep it unchanged while experimenting.

**Q4. Which data may experiments use?**

The train split is for parameter fitting, the crowdsourced validation split is for model selection and tuning, and expert min1/min2/min3 splits are sealed final evaluation only. Expert results must not select architectures, hyperparameters, stopping points, or augmentations.

**Q5. What kind of change comes first?**

Perform error analysis first, then choose the smallest intervention supported by evidence. Candidate axes include loss/class weighting, augmentation, architecture, label quality, and sampling.

**Q6. How will experiments remain interpretable?**

Change one meaningful variable per experiment. Keep the dataset, manifests, seed, input size, optimizer, schedule, and evaluation code fixed unless explicitly declared as the research variable.

**Q7. What is the minimum experiment record?**

Save the exact configuration, source revision, random seed, dataset and split-manifest identities, environment and device information, training and validation metrics by epoch, final checkpoint hash, per-class metrics, confusion matrix, and hypothesis/result notes.

**Q8. Where should implementation happen?**

Use a separate Git branch or clearly named experiment directory. Preserve the baseline workflow, add focused tests, run the smoke configuration first, then run the smallest useful training experiment.

**Q9. What is the first concrete study?**

Read the reproduction protocol and model contract; inspect baseline errors and big-rock confusion; write one hypothesis; define one measurable intervention; run a small controlled experiment; compare it against the frozen baseline; and record the result, including a negative result.

**Q10. What exactly is the 40-epoch control?**

Use the best validation-mIoU checkpoint from the completed 40-epoch run as the control. The 40-epoch label describes the training budget, not necessarily the selected checkpoint epoch. Record the actual selected epoch.

**Q11. What is the analysis order?**

First analyze only crowdsourced validation. Lock the intervention and configuration. Then evaluate the frozen baseline and locked candidate separately on expert min1/min2/min3 without changing the candidate.

**Q12. What is the first analysis evidence?**

Generate per-class IoU and recall, raw and row-normalized confusion matrices, bedrock-to-big-rock and big-rock-to-bedrock counts and rates, representative error examples, and error breakdowns by image or sequence where available.

**Q13. What is the first intervention boundary?**

Keep the first candidate as a four-class model. Test class weighting, sampling, or label handling separately; do not combine them.

**Q14. What is the acceptance threshold?**

Require at least `+0.03` absolute big-rock IoU, while overall mIoU regresses no more than `-0.01`, each non-big-rock class IoU no more than `-0.02`, and pixel accuracy no more than `-0.01`.

**Q15. Is one seed enough?**

No. Use seed 42 for the initial controlled test. Repeat promising changes across predetermined seeds, such as 7, 42, 123, and 2026, before making a research claim.

## Questions 16-31: Baseline, Evaluation, and Future Architecture

**Q16. What is the epoch-40 artifact contract?**

Use the run's `best_val_miou.pth`, not automatically `last.pth`. Record its path, selected epoch, 40-epoch budget, global step, exact configuration or configuration hash, source revision, SHA-256, and validation metrics.

**Q17. What is the analysis dataset?**

Use crowdsourced validation for diagnosis and iteration. Evaluate expert min1/min2/min3 only after the candidate is locked.

**Q18. What does “three classes” mean?**

The future terrain model predicts soil, sand, and bedrock. Big rock is a separate object/hazard task, not automatically a terrain class.

**Q19. What is the dedicated big-rock model's role?**

Use a two-stage conceptual system: an independent three-class terrain model and an independent big-rock model. Do not commit to Mask R-CNN until labels support instance-level targets.

**Q20. What is the first deliverable?**

Produce the baseline confusion report, write the hypothesis it supports, and create one isolated experiment configuration. Analysis comes before training changes.

**Q21. What is the annotation prerequisite for Mask R-CNN?**

Inspect mask topology and annotation format first. If masks contain only semantic class pixels, use binary semantic segmentation before instance segmentation.

**Q22. What is the analysis completion condition?**

Analysis is complete only when all required metrics, confusion summaries, top-10 overlays, image/sequence breakdowns, and metadata are saved reproducibly.

**Q23. What is the decision rule?**

Missed rocks select class weighting or loss rebalancing; sparse examples select sampling; ambiguous boundaries select label inspection; sequence-specific failures select domain-shift investigation; mixed evidence selects no intervention yet.

**Q24. What is the immediate action while training runs?**

Prepare and verify the analysis procedure, expected artifact path, results naming, baseline evaluation command, hypothesis template, and acceptance criteria. Do not tune against the old checkpoint if the 40-epoch run is the intended control.

**Q25. What is the analysis order before expert evaluation?**

Analyze crowdsourced validation first. Lock the intervention and candidate configuration. Then run baseline and candidate on expert splits without revising either.

**Q26. What metadata identifies the control?**

Record checkpoint path, selected epoch, total 40-epoch budget, global step, exact config or hash, source revision, checkpoint SHA-256, and validation metrics.

**Q27. What are the regression tolerances?**

Use the provisional screening limits: big-rock IoU at least `+0.03`, overall mIoU no worse than `-0.01`, each non-big-rock IoU no worse than `-0.02`, and pixel accuracy no worse than `-0.01`.

**Q28. What do current annotations support?**

Current masks support semantic class evaluation, not instance-level rock detection. Individual rocks need distinct instance identities before Mask R-CNN can be evaluated fairly.

**Q29. What is the conditional hypothesis?**

If validation shows systematic missed big-rock pixels and strong underrepresentation, increasing only the big-rock loss weight should improve big-rock IoU by at least `0.03` without violating the regression limits. If errors are ambiguous boundaries, inspect labels instead.

**Q30. What is the first model scope?**

Complete and analyze the four-class baseline first. The three-class model and dedicated rock model are later studies.

**Q31. What becomes immutable before expert evaluation?**

Lock the candidate checkpoint, code revision, configuration, seed list, preprocessing, class mapping, stopping rule, and validation-based selection decision.

## Questions 32-51: Artifacts and Protocol Closure

**Q32. Where should the new baseline live?**

Keep the epoch-25 checkpoint untouched as historical evidence. The new 40-epoch-budget artifact is identified by its authoritative run output.

**Q33. How many examples are inspected?**

Inspect the top 10 examples for missed big-rock pixels, false positives, bedrock/big-rock boundary confusion, small or fragmented rock regions, and sequence-level failures.

**Q34. How should future terrain labels treat big rock?**

Do not automatically map big rock to bedrock unless dataset semantics establish that the underlying terrain is bedrock. Provisionally map big-rock pixels to `ignore`, because the underlying terrain is occluded.

**Q35. Are thresholds proof of generalization?**

No. They are screening thresholds. Promising candidates still require multiple seeds and locked expert evaluation.

**Q36. What may be implemented before the checkpoint arrives?**

Only the analysis path: validation evaluation, overlays, failure categorization, and reporting. Do not add a model or loss option yet.

**Q37. Is instance annotation a separate project?**

Yes. Record instance-level big-rock labeling as a later research track. Existing semantic labels support binary segmentation but not fair instance evaluation.

**Q38. What is the artifact preservation rule?**

Keep epoch 25 at its current frozen path and label it historical reference evidence. Do not overwrite it with the 40-epoch artifact.

**Q39. Is metadata part of the artifact?**

Yes. Metadata is mandatory evidence, not optional notes.

**Q40. Where do reports and figures go?**

Write validation-only reports and confusion summaries under `expert-evaluation`, using separate validation and expert subdirectories. Save top-10 error overlays under `figures`.

**Q41. What does “locked candidate” mean?**

Record the hypothesis, changed variable, unchanged controls, configuration, seed list, selected checkpoint, validation results, and acceptance decision in one candidate record. Expert evaluation must not alter it.

**Q42. What is the execution order?**

Finish the 40-epoch run; verify and hash its best checkpoint; record metadata; analyze crowdsourced validation; inspect errors; choose one intervention; train and screen the candidate; lock it; evaluate both on expert splits; then begin the later architecture studies.

**Q43. What is the provisional terrain mapping?**

Preserve original masks. Provisionally map big rock to `ignore` in the three-class terrain target, not bedrock, unless future dataset evidence justifies another mapping.

**Q44. What is the 40-epoch artifact path?**

The completed run's checkpoint is:

```text
<output_root>/runs/paper-deeplabv3plus-kaggle-p100/checkpoints/best_val_miou.pth
```

The concrete Kaggle path is specified in Q76.

**Q45. Should validation and expert outputs be separated?**

Yes. Use separate subdirectories so validation evidence cannot be confused with held-out expert evidence.

**Q46. What is the intended downstream system?**

The three-class model provides dense soil, sand, and bedrock terrain labels. A separate big-rock instance model detects physical hazards and may later support navigation and scientific morphology analysis. Terrain type and obstacle presence are separate perception tasks.

**Q47. How is the dedicated binary model evaluated?**

On full images, independently of the three-class model, against the original big-rock semantic mask over the full valid image area. ROI-restricted or cascaded inference is a later secondary experiment.

**Q48. Which seeds confirm a result?**

Use at least three additional predetermined seeds, with seed 42 included for continuity. Do not add or remove seeds after seeing results.

**Q49. What makes analysis reproducible?**

Save raw results, metrics, confusion matrices, examples, overlays, rankings, manifest identity, checkpoint identity, and metadata. Prose alone is insufficient.

**Q50. How are observed patterns mapped to actions?**

Use the written decision rule before inspecting results: missed-rock dominant selects weighting; sparse-example dominant selects sampling; boundary-confusion dominant selects label audit; domain-specific failures select domain-shift investigation; false positives select confusing-terrain analysis; label problems select annotation audit; mixed evidence selects no tuning.

**Q51. What is the final pre-analysis sequence?**

Finish the 40-epoch run, select and hash its best checkpoint, preserve epoch 25, generate validation analysis, select one intervention, screen it, lock it, and only then run sealed expert evaluation.

## Questions 52-67: Detailed Model and Evaluation Contract

**Q52. What is the primary big-rock target?**

The first dedicated model is full-image binary semantic segmentation using existing semantic masks. Instance segmentation, detection, and stereo geometry are deferred.

**Q53. What is the exact three-class relabeling?**

Soil remains soil, sand remains sand, bedrock remains bedrock, big rock becomes `ignore`, and original `255` remains `ignore`. Preserve original four-class masks unchanged.

**Q54. What is the primary binary-model metric?**

Big-rock IoU is primary. Also report precision, recall, confusion counts, and per-image/per-sequence performance. Pixel accuracy may be reported but must not drive conclusions.

**Q55. What must the output-root metadata contain?**

Record the absolute run root, Kaggle dataset/output identity, source revision or commit, checkpoint filename, and checkpoint hash. Do not invent a path or identity that has not been recorded.

**Q56. How is the historical baseline preserved?**

Keep epoch 25 immutable at its current frozen path. Evaluate the 40-epoch run directly from its authoritative output location.

**Q57. What if the 40-epoch run is worse than epoch 25?**

The 40-epoch run's best-validation checkpoint remains the budget-matched experimental control. Report it separately from epoch-25 historical evidence.

**Q58. What is the stereo boundary?**

Stereo-derived height, width, range, and position are future work unless calibrated stereo pairs and a validated reconstruction pipeline demonstrably exist.

**Q59. What is the first implementation after analysis?**

Implement reproducible validation analysis, overlays, failure categorization, and reporting before modifying model, loss, weighting, or sampling.

**Q60. What is the complete research contract?**

Preserve epoch 25; finish the 40-epoch run; select its best validation checkpoint; analyze validation; select one intervention; screen it; lock it; evaluate sealed experts; then begin the independent terrain and rock studies.

**Q61. What does the terrain model mean at rock pixels?**

It classifies exposed terrain where terrain is observable. Original big-rock pixels are excluded because the underlying terrain cannot reliably be inferred. Report the ignored-rock fraction and changed evaluation domain.

**Q62. What data is required for instance modeling?**

Annotations must assign a distinct identity to each qualifying rock and define instance semantics, touching/overlap/occlusion/truncation policy, quality control, splits, and instance metrics.

**Q63. What is the immediate four-class intervention?**

Modify exactly one intervention category selected by the predeclared rule while keeping the four-class target and all other conditions fixed.

**Q64. What is the candidate comparison reference?**

Measure the `+0.03` big-rock IoU threshold against the 40-epoch-budget best-validation control, not the epoch-25 historical checkpoint.

**Q65. How are future models evaluated?**

The terrain model excludes original `255` and big-rock pixels. The binary model treats big-rock as positive, soil/sand/bedrock as negative, and `255` as excluded over the full valid image.

**Q66. What is the report interpretation boundary?**

Separate observed evidence, interpretation, and recommendation. Raw results must be stored independently from narrative conclusions.

**Q67. What is the complete contract at this stage?**

The epoch-25 checkpoint is historical; the 40-epoch checkpoint is the budget-matched control; validation comes before expert data; one four-class intervention is selected; thresholds screen it; the candidate is locked; expert evaluation follows; three-class and binary studies come later.

## Questions 68-76: Final Reproducibility Rules

**Q68. When is a failure mode dominant?**

A category is dominant if it accounts for at least 50% of categorized big-rock error pixels or contains at least twice the error mass of the next-largest category. Record affected pixel and image counts. Otherwise classify evidence as mixed/inconclusive and do not change the model.

**Q69. How are top-10 examples selected?**

Missed rocks rank by descending false-negative pixels; false positives by descending false-positive pixels; boundary confusion by descending boundary disagreement; fragmentation by descending predefined fragmentation score; and sequence failures by ascending aggregate sequence IoU. Use image ID as the tie-breaker and record all ranking fields.

**Q70. How are questionable labels handled?**

Preserve the original mask, record image and region details, obtain an explicit review decision, store approved corrections as a new versioned annotation set, and treat corrected-label experiments as a separate dataset version.

**Q71. What must be identical in the control/candidate validation comparison?**

Use identical manifests, preprocessing, resize/crop behavior, ignore handling, valid-pixel definition, class mapping, metrics, inference procedure, and checkpoint-selection policy. The declared intervention is the only intended difference.

**Q72. What does expert evaluation compare?**

Run the locked 40-epoch control and locked candidate through the same expert protocol on min1, min2, and min3. Report them side by side. Do not use expert results to revise either model.

**Q73. What is the experiment naming convention?**

Use `<task>_<intervention>_seed<seed>_ep<budget>`, for example `4class_control_seed42_ep40`, `4class_classweight_seed42_ep40`, and `4class_oversample_seed42_ep40`. Avoid ambiguous names such as `final` or `new_model`.

**Q74. What is the first concrete implementation slice?**

Implement deterministic validation analysis and artifact generation before any new training or model option. It must produce checkpoint identity, source revision, manifest identity/hash, aggregate and per-class metrics, confusion data, per-image/per-sequence results, categorized errors, deterministic top-10 examples, overlays, dominance result, interpretation, and recommendation records.

**Q75. What must be frozen before inspecting results?**

Freeze dominance thresholds, top-10 ranking rules, output-root and artifact identity, experiment naming, expert comparison protocol, and the decision rule.

**Q76. What is the concrete output root?**

The completed baseline checkpoint will be at:

```text
/kaggle/working/ai4mars-paper-reproduction/runs/paper-deeplabv3plus-kaggle-p100/checkpoints/best_val_miou.pth
```

No Kaggle Dataset/output-version identity is currently configured, so the completed notebook output must be saved as an explicit Kaggle version. Record that version identity with the baseline metadata.

## Final Execution Order

1. Finish the 40-epoch-budget run.
2. Save the notebook output as an explicit Kaggle version.
3. Select and hash `best_val_miou.pth` and record its actual epoch.
4. Preserve epoch 25 as historical evidence.
5. Analyze crowdsourced validation only.
6. Inspect deterministic top-10 error examples.
7. Apply the predeclared failure-mode rule.
8. Train one isolated four-class candidate.
9. Screen it against the validation thresholds.
10. Lock its checkpoint, configuration, metadata, and hash.
11. Evaluate the locked control and candidate on expert min1/min2/min3.
12. Report expert results as held-out evidence.
13. Begin the three-class terrain and independent full-image binary big-rock studies.
14. Defer instance segmentation and stereo geometry until the required data and protocols exist.
