# Rock Instance Annotation Protocol

## Status And Scope

This is an annotation-development protocol for testing the rock-instance hypothesis. It is not established ground truth and does not convert AI4Mars semantic `big_rock` connected components into true instances automatically. The initial detector class is conceptually `rock`; physical size and hazard status are deferred to verified geometry.

## Unit Of Annotation

An accepted discrete rock instance is a visually separable, spatially localized rock object whose visible boundary can be distinguished from surrounding terrain with sufficient confidence for review. Annotate the visible object extent, not an inferred hidden full volume.

Continuous exposed substrate is `continuous_bedrock`, not a rock instance, even when the semantic mask uses Big Rock. A candidate connected component may contain zero, one, or many rock objects.

## Review Statuses

| Status | Meaning | Annotation action |
| --- | --- | --- |
| `candidate_unreviewed` | Automated semantic candidate; no human decision yet | Review; do not train from it |
| `accepted_discrete_instance` | One discrete, separable rock | Create one reviewed instance mask/boundary |
| `merged_multiple_objects` | Candidate visually merges two or more rocks | Split into separately reviewed instances where boundary evidence supports it; otherwise mark uncertain |
| `fragmented_single_object` | Candidate has multiple semantic fragments but is one rock | Merge fragments into one reviewed visible-object annotation if defensible |
| `continuous_bedrock` | Exposed substrate or continuous outcrop | No rock instance; retain disposition |
| `truncated` | Object enters an image border | Annotate visible portion and record truncation |
| `occluded` | Object is partly blocked by rover hardware, terrain, shadow, or another object | Annotate only defensible visible extent and record occlusion |
| `uncertain` | Rock/bedrock boundary or object identity cannot be resolved reliably | Do not force an instance; retain for adjudication |
| `rejected_noise` | Tiny artifact, isolated label noise, or non-rock region | No instance |

Statuses may be combined with flags such as `truncated` and `occluded`. `accepted_discrete_instance` requires a reviewer decision; it is never inferred from area, connectedness, or bounding-box shape.

## Difficult Cases

- Touching or overlapping rocks: draw separate instances only where an object boundary is visually defensible. Otherwise use `merged_multiple_objects` or `uncertain`, never arbitrary watershed splitting.
- Partially buried rocks: annotate the visible exposed rock as one object if it remains separable; record partial burial.
- Image-border objects: retain the visible portion and mark `truncated`; do not reject solely for truncation.
- Occluded rocks: retain a visible-object annotation only when enough object evidence remains; do not hallucinate occluded extent.
- Rock/bedrock transitions: prefer `continuous_bedrock` or `uncertain` when a discrete object boundary is not credible.
- Tiny components: treat as review triggers, not automatic objects. Use `rejected_noise`, `uncertain`, or an accepted instance only after visual review.

## Review Record

For every candidate reviewed, retain stable source image ID, sequence ID, candidate component IDs, reviewer and review date, primary status, flags, visible-instance count, polygon/mask references if accepted, confidence, and a concise rationale. Keep source semantic masks unchanged.

## Quality Control

Double-review an initial calibration subset across all statuses, including boundary and occlusion cases. Adjudicate disagreements before expanding annotation. Report agreement by decision type and document any protocol revision. Do not use expert-test imagery to develop this protocol, choose component thresholds, or select annotator examples.

## Calibration And Freeze Workflow

The required workflow is:

1. Generate train/val-only semantic candidates and select the deterministic calibration subset.
2. Review the calibration images using the versioned review state. Record accepted, rejected, split-required, merge-required, uncertain, deferred, truncated, occluded, and rationale decisions rather than forcing a label.
3. Revise this protocol if calibration exposes systematic disagreement, ambiguous boundary rules, or missing status definitions. Record the revision and revisit affected calibration decisions.
4. Review the remaining pilot candidates only after the calibration protocol is stable. A single researcher must record uncertain/deferred decisions and rationale; this is not multi-expert consensus.
5. Validate the reviewed export, inspect the quality report, and freeze it as `rock_instance_pilot_v1` with schema version, candidate-manifest hash, reviewer metadata, and review-version provenance.
6. Only then create a controlled RGB-only Mask R-CNN experiment.

`accepted` is the only status eligible for future detector targets and requires `discrete_rock=true` plus reviewer-supplied visible-object polygon geometry. `truncated` and `occluded` are flags, never physical-hazard labels. `split_required`, `merge_required`, `uncertain`, and `deferred` are explicit blockers for detector-target creation until resolved.
