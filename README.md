# AI4Mars Rock Perception

Martian terrain segmentation and instance-level rock detection from rover imagery.

## Introduction

This project uses semantic segmentation as a controlled baseline for development of a separate rock-instance classifier. The baseline is intended to characterize the limitations of pixel-wise `big_rock` classification and provide a reference for object-level rock detection.

The planned system separates broad terrain context from discrete rock hazards:

- semantic segmentation: `soil`, `bedrock`, and `sand`;
- instance-level detection or segmentation: individual rocks;
- stereo geometry: rock distance and physical dimensions.

## Dataset

Experiments use the merged AI4Mars 0.6 release and are currently restricted to the Mars Science Laboratory NAVCAM subset.

| Value | Class |
|---:|---|
| 0 | Soil |
| 1 | Bedrock |
| 2 | Sand |
| 3 | Big Rock |
| 255 | Ignore |

Dataset files are not included in this repository.

Inputs are padded internally to satisfy encoder stride requirements and cropped back to `513 × 513` before loss and metric computation.

## Preliminary Results

A three-epoch development run reached a mean IoU of `0.7837`.

| Class | IoU |
|---|---:|
| Soil | 0.9256 |
| Bedrock | 0.9359 |
| Sand | 0.8822 |
| Big Rock | 0.3911 |

Big Rock remains the principal failure mode and is predominantly confused with Bedrock. These are development-validation results, not final expert-test results.

### Reference

R. M. Swan et al., “AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars,” CVPR Workshops, 2021.
