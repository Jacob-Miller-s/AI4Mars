# Sprint 0 Findings

This report records a train/val-only execution against the mounted AI4Mars merged 0.6 release. Generated CSV/JSON/PNG outputs remain under `outputs/sprint0/` and are not source-controlled dataset artifacts.

## Stereo And Geometry Inventory

- development images inspected: 16,055 (12,944 train; 3,111 val)
- image files present: 16,055
- confirmed stereo mates: 0
- confirmed metric depth products: 0
- sufficient calibration/geometry metadata: 0
- unresolved stereo/geometry relationships: 16,055
- documented `rng-30m` binary range-validity masks present: 16,054

The release documentation says RNG-30M masks ranges beyond 30 m and points to external PDS range products. The masks are not metric depth, camera calibration, or confirmed stereo pairing. Sprint 0 therefore does not support stereo-derived physical geometry from the mounted release alone.

## Semantic Candidate Audit

The audit used 8-connected components of NAV semantic class 3 (`big_rock`) only as manual-review candidates.

- candidate-bearing images: 2,226
- candidate components: 7,707 (6,221 train; 1,486 val)
- component area: minimum 1 px, median 207 px, mean 11,187.959 px, maximum 744,370 px
- images queued for manual review: 1,766
- ambiguity signals: 1,320 multi-component images, 875 fragmentation proxies, 826 nearby separate-component cases, 1,045 unusual-aspect-ratio cases, 435 very-large components, 916 tiny-component cases, 86 Bedrock/Big Rock boundary cases, and 4 border-truncation candidates

The dispersion and ambiguity counts rule out treating components as automatically valid instance ground truth. A substantial review and adjudication phase is required.

## Proposed Pilot

The deterministic seed-42 proposal contains 150 unreviewed candidate images from 150 distinct acquisition sequences (119 train, 31 val). It contains 30 isolated candidates, 120 multi-region cases, 44 Bedrock-boundary cases, 93 very-large cases, 137 tiny cases, 3 available border cases under the sequence-diversity cap, and 150 RGB-only/geometry-unresolved cases.

Estimated burden: review all 150 pilot images, with the 1,766 ambiguity-flagged images forming the follow-on queue. The exact number of final instances and adjudication time are unknown until visual review; connected-component count is not an estimate of the final instance count.

## Sprint 1 Gate

The repository is ready for an RGB-only detector pilot only after the 150-image candidate manifest is manually reviewed and a reviewed instance annotation export is versioned. The recommended entry point is a controlled RGB-only Mask R-CNN with ResNet-50-FPN, compared on object-centric reviewed labels against a separately documented semantic-component adapter. Stereo/geometry is blocked pending explicit pair, calibration, and metric range provenance.
