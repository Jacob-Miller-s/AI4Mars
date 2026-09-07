# Rock Instance Calibration Closure - v2.3

## Final Status

`CALIBRATION_PROTOCOL_STATUS = FROZEN`

`CALIBRATION_PROTOCOL_VERSION = v2.3-calibration-final`

## Calibration Record

1. The initial 24-image calibration was exploratory and later found insufficiently complete per semantic candidate component.
2. The corrected component-complete calibration reviewed 24 images and covered 173/173 candidate components.
3. An isolated repeat review was conducted as a separate state; agreement evidence was used descriptively and did not declare either pass ground truth.
4. Object-identity clarification separated semantic Big Rock components from physical rock instances.
5. Visible-extent clarification established the full defensible visible image-plane object rule, excluding shadow, hidden geometry, and continuous Bedrock.
6. The final whole-object clarification preserved component 8 as an accepted object but showed no reproducible RGB-supported mask boundary.
7. That result is recorded as one `boundary_indeterminate` exclusion and remains distinct from the 5 historical image-level `uncertain` exclusions.
8. v2.3 freezes the final distinction: accepted identity with a reproducible boundary yields a polygon; accepted identity without one excludes the entire image from ordinary Mask R-CNN targets without becoming background.
9. Known limitations: calibration is intentionally difficult-case enriched, uses RGB only, and does not establish physical dimensions, range, stereo/depth, or hazard status.
10. Early calibration acceptance rates are not representative full-dataset statistics and must not be extrapolated as such.

## Frozen Provenance

- Protocol SHA-256: `5db68378e52bfc9e32fa1fea3c10338606a92ea785da3a5ebf8eb4a190495fc4`
- Freeze artifact SHA-256: `07dc16a6781fd4d6e0a47869accae3f181db48184a9d31b4138a054a525c9da9`
- Calibration closure SHA-256: `d1ac5f02cae416128712ed32f443935269ba3c51d07ffb246aee1deaf3038f2d`
- Boundary-indeterminate ledger SHA-256: `faf373f8a2ae3e600d9f05a0eaafb1d91f161eaaa40736a5a097aacd8ea59069`
- Freeze timestamp (UTC): `2026-08-08T07:56:21.530449+00:00`

The protocol was frozen before expansion to the remaining 126-image production review. This report authorizes neither target export nor model training, and it does not freeze `rock_instance_pilot_v1`.
