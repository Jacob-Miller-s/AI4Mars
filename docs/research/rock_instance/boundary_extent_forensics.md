# Boundary Extent Forensics

## Guardrails

This is evidence preparation for a three-object, fixed-identity boundary redraw. Primary, repeat, and v2.1 annotations remain immutable. The prior polygons are rendered only in forensic figures and are hidden in the redraw UI.

## Findings

| Target | Evidence-supported extent pattern | Boundary classification | Escalation implication |
| --- | --- | --- | --- |
| `NLB_463551084EDR_F0411534NCAM00385M1`, component 4 | Primary/repeat polygons are elongated (`[184,261,112,43]`, `[206,262,88,41]`); v2.1 is a compact proposal-core polygon (`[176,296,30,19]`) with IoU `0.0263` to primary. | Candidate-core/protruding-face trace versus broader visible feature; not ordinary contour variance. | If the nonbinding proposal does not locate the same visible object in redraw, record identity escalation rather than relabeling. |
| `NLB_483955685EDR_F0470598NCAM00320M1`, component 8 | v2.1 (`[816,204,69,43]`) closely follows the candidate box (`[817,205,66,40]`), while primary/repeat span larger extents (`[790,189,93,81]`, `[797,199,81,63]`). The image includes rover hardware and shadow nearby. | Semantic-proposal core versus broader material/terrain interpretation; occlusion and shadow must be excluded explicitly. | Redraw the visible rock only; escalate only if the fixed rock cannot be located separately from the nearby hardware/terrain. |
| `NLB_548252623EDR_F0631150NCAM00312M1`, component 3 | Primary/repeat trace a broad elongated region (`[314,75,335,149]`, `[310,73,336,156]`); v2.1 traces a compact proposal-core (`[613,113,57,56]`) with IoU `0.0008`/`0.0025`. | Proposal-core/protruding-face trace versus broad visible-surface interpretation; not ordinary drawing variance. | If the proposal does not reliably locate the fixed accepted object, use identity escalation instead of silently treating the mismatch as a boundary issue. |

Forensic overlays are written when the isolated package is prepared under [calibration_boundary_clarification_v2](../../../artifacts/rock_instance/calibration_boundary_clarification_v2). Each shows RGB, terrain context, exact semantic proposal pixels, and each historic polygon.

## Tooling Bias

The existing general reviewer uses full-image coordinates and does not clip polygons to a candidate box. It does, however, render the semantic `big_rock` overlay and candidate boxes continuously, with no visibility toggle. That can visually anchor a reviewer to the proposal core. The boundary redraw UI corrects this narrowly: full-resolution RGB is the drawing surface, prior polygons are absent, semantic proposals are hidden by default, and proposal boxes can be shown only as a labeled nonbinding reference. Neighboring proposal boxes remain available through that optional view.

## Human Workflow

Prepare the empty package once:

```powershell
python -m src.rock_instance.boundary_review --prepare `
  --primary-state-path artifacts/rock_instance/calibration_resolved_v2/review_state.json `
  --repeat-state-path artifacts/rock_instance/calibration_repeat_v1/review_state.json `
  --v21-state-path artifacts/rock_instance/calibration_clarification_v1/review_state.json `
  --component-candidates-csv artifacts/rock_instance/calibration_clarification_v1/big_rock_component_candidates.csv `
  --target-manifest research/rock_instance/boundary_clarification_v2.2-targets.csv `
  --proposed-protocol-path research/rock_instance/annotation_protocol_v2.2-visible-extent-clarified-proposed.md `
  --dataset-root data/raw/ai4mars/ai4mars-dataset-merged-0.6 `
  --output-dir artifacts/rock_instance/calibration_boundary_clarification_v2
```

Then redraw exactly the three targets:

```powershell
python -m src.rock_instance.boundary_review `
  --state-path artifacts/rock_instance/calibration_boundary_clarification_v2/review_state.json `
  --dataset-root data/raw/ai4mars/ai4mars-dataset-merged-0.6 `
  --interactive --reviewer single_researcher
```

The reviewer answers only: **What visible pixels belong to this already-accepted rock object?** Use the Matplotlib toolbar to pan and zoom. Use `Show nonbinding proposals` only to locate the proposal; do not trace it. Save a polygon and finish each target, or use `Escalate identity` with a concrete note. Do not review any other image.

## Post-Redraw Recheck

Compare each new mask with primary, repeat, and v2.1 while keeping none as ground truth. Report area, bbox, pairwise overlap, visible-surface reasoning, and whether any difference is ordinary contour precision or still a conceptual extent disagreement. Return `FREEZE` only after all three are redrawn without identity escalation and follow this same visible-extent target.

Run the fail-closed geometry report only after all three targets are completed without escalation:

```powershell
python -m src.rock_instance.boundary_consistency `
  --primary-state-path artifacts/rock_instance/calibration_resolved_v2/review_state.json `
  --repeat-state-path artifacts/rock_instance/calibration_repeat_v1/review_state.json `
  --v21-state-path artifacts/rock_instance/calibration_clarification_v1/review_state.json `
  --boundary-state-path artifacts/rock_instance/calibration_boundary_clarification_v2/review_state.json `
  --output-dir artifacts/rock_instance/calibration_boundary_clarification_v2/consistency `
  --markdown-path docs/research/rock_instance/v2.2_boundary_consistency.md
```