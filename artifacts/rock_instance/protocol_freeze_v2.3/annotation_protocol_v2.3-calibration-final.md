# Rock Instance Annotation Protocol v2.3 - Calibration Final Candidate

Protocol identifier: `v2.3-calibration-final`

## Status

This is the final calibration candidate for human approval. It is not activated by its existence, does not alter historic annotations or redraws, and does not authorize pilot expansion, target export, or model training.

## Inherited Rules

This revision preserves the v2.1 object-identity clarification and the v2.2/v2.2.1 visible-image-plane, whole-object, continuity, shadow, occlusion, Bedrock, and conservative-ambiguity rules. Semantic components remain nonbinding review/provenance references.

## Boundary-Indeterminate Exclusion

A reviewer may determine that a discrete physical rock is visibly present while also determining that available RGB evidence does not support a sufficiently defensible and reproducible instance boundary.

In this outcome, preserve `object_identity = accepted`, record `boundary_status = indeterminate`, retain source-component provenance and reason/evidence, and do not force an arbitrary polygon. Do not create a positive Mask R-CNN mask target and do not silently convert the region to background. Exclude the affected image from ordinary Mask R-CNN target generation for the current pilot unless a separately validated ignore-region mechanism is introduced.

This is distinct from `uncertain`:

- `uncertain`: the RGB evidence does not support a defensible determination that a discrete rock instance exists.
- `boundary_indeterminate`: a discrete rock is accepted as present, but its visible image-plane extent is not reproducible enough for instance-mask ground truth.

The outcomes must not be collapsed.

## Scope and Versioning

The calibration conclusion is limited to the reviewed evidence and its recorded exclusions. Future protocol changes require a new explicit version and a new evidence-bound calibration review; they do not modify this protocol or historic artifacts.