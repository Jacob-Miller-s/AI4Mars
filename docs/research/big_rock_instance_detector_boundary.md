# Future Big-Rock Instance Extension

This branch does not implement instance detection. The future branch is `feat/big-rock-instance-detector` and consumes a stable semantic prediction contract rather than changing the reproduction training path.

The proposed direction is three-class semantic segmentation for soil, bedrock, and sand plus separate rock-instance detection. Stereo or depth should estimate physical dimensions, and hazard-relevant physical size should determine the big-rock designation. Semantic-instance fusion may later become a panoptic-style output.

AI4Mars semantic masks do not provide rock-instance identities. Candidate future work includes connected-component pseudo-instances, a manually corrected instance evaluation subset, object-level detection/size metrics, depth integration, and a fusion interface accepting semantic logits, detector boxes or masks, stereo/depth, calibration, and confidence. None of these are valid substitutes for semantic reproduction metrics.