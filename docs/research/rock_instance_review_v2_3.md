# Rock-instance production review v2.3

Rock-instance annotation is a research phase separate from the frozen semantic-reproduction baseline. It does not change the baseline configuration, outputs, or evaluation path.

The restored local Matplotlib reviewer is limited to the approved v2.3 production package:

- 126 remaining pilot images;
- 814 candidate components;
- 0/126 images reviewed in the prepared state.

Every load, save, and review mutation fails closed unless the protocol freeze, calibration state, source pilot, candidate manifests, and boundary-indeterminate exclusion ledger match their approved SHA-256 provenance. The protocol and calibration evidence remain immutable under `artifacts/rock_instance/`; methodological changes require a new protocol version.

Launch the production reviewer from the repository root with the separately obtained AI4Mars dataset:

```bash
python -m src.rock_instance.production_review review \
  --dataset-root <ai4mars-dataset-root> \
  --reviewer <reviewer-id>
```

The reviewer saves state atomically and resumes the first unfinished production image. Accepted instances require polygons. Merge and split records retain source-component and resulting-instance lineage. `uncertain` remains distinct from `boundary_indeterminate`; the latter preserves accepted object identity without an arbitrary polygon and excludes the whole image from ordinary instance-mask targets.

To regenerate the progress summary after a review session:

```bash
python -m src.rock_instance.production_review progress \
  --state-path artifacts/rock_instance/production_review_v2.3/review_state.json \
  --output-json artifacts/rock_instance/production_review_v2.3/review_progress.json
```

No dashboard, model-training path, or earlier experimental reviewer is part of this restoration.
