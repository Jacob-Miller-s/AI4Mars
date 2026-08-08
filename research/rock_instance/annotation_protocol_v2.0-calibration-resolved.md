# Rock Instance Annotation Protocol v2.0 - Calibration Resolved

## Status

This protocol implements the approved calibration-resolution policy. It is a draft operating protocol, not a frozen dataset specification. The human reviewer must complete corrected calibration review and the isolated repeat review before any protocol freeze, remaining-pilot review, detector training, or Mask R-CNN target export.

`v1.0-initial` is preserved unchanged in `annotation_protocol_v1.0-initial.md`. The immutable initial calibration snapshot is evidence, not a source to edit or overwrite.

## Review Unit And Provenance

The semantic Big Rock connected components are review references only. A physical rock instance may have zero, one, or multiple candidate-component sources. Every terminal annotation records plural `source_candidate_component_ids`; legacy scalar `source_candidate_component_id` remains a compatibility alias for one source component.

Each corrected calibration image has an explicit expected candidate-component list. It can be marked reviewed only when every expected component is covered by either:

1. a direct terminal disposition: `accepted`, `rejected_bedrock`, `rejected_noise`, or `uncertain`; or
2. a human-authored split or merge resolution record.

Candidate connectivity, pixel area, adjacency, and bounding-box shape never decide a split or merge.

## Split And Merge Resolution

`split_required` and `merge_required` are initial-review findings, not terminal instance targets. A resolution record is append-only and contains an ID, image and sequence provenance, source component IDs, linked immutable initial-decision IDs, linked corrected terminal annotation IDs, resolution type, and reviewer rationale.

- Split: one source component can produce two or more accepted visible-object children only when the human records defensible boundaries. Each child retains the parent source component ID.
- Merge: multiple source components can produce exactly one accepted visible-object instance only when the human confirms one physical rock. The merged instance retains every contributing source component ID.
- A split or merge may resolve to one terminal non-accepted disposition when the reviewer concludes no defensible instance should be created.

No tooling creates a resolution record, child instance, merged instance, or polygon automatically.

## Terminal Uncertainty

`uncertain` is terminal for this calibration protocol. It retains the reviewer decision and rationale, but it is neither a positive object target nor a negative/background example. An image containing any terminal uncertain annotation is excluded as a whole from ordinary Mask R-CNN target conversion. It must not be partially converted by silently ignoring the uncertain region.

## Candidate-Independent Observation

Each corrected calibration image records whether an obvious rock was observed outside all semantic candidate components. This is an image-level observation and explanatory note only. It never creates an inferred instance, polygon, or detector target.

## Corrected Calibration Workflow

1. Initialize a new `calibration_resolved_v2` state from copied train/val manifests, the v2 protocol, and immutable initial-snapshot path/hash provenance. Do not modify `calibration_initial_v1`.
2. Review every expected candidate component, saving direct terminal dispositions or human-authored resolution records with links to initial decisions.
3. Finish an image only after strict component coverage passes. Any unresolved component blocks completion.
4. After all corrected calibration images are final, select the repeat subset with seed `42` and rank key `SHA256("calibration-repeat-v1:42:" + image_id)`.
5. Create an isolated empty repeat-review state. It must not share mutable annotations or resolution records with the primary corrected state.
6. Compare the repeat-review decisions and document discrepancies before proposing a protocol freeze.

The implementation does not perform these human actions and does not authorize pilot expansion, dataset freeze, Mask R-CNN training, semantic-baseline changes, or expert-split access.