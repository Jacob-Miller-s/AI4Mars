# AI4Mars Rock Perception

Martian terrain segmentation and instance-level rock detection from rover imagery.

## Introduction

The objective of this work is to develop an instance-level perception method for identifying discrete
rover-scale rocks in Martian surface imagery. A semantic segmentation baseline is first established
using the AI4Mars dataset as a necessary reference for understanding the limitations of pixel-wise
terrain classification. In particular, the baseline is used to measure the degree to which Big Rock
is confused with Bedrock and other terrain classes. These observations guide the subsequent
development of a separate rock instance classifier intended to identify individual rock objects,
preserve their boundaries, and eventually incorporate stereo-derived estimates of physical size.
## Dataset

Experiments use the merged AI4Mars 0.6 release and are currently restricted to the Mars Science Laboratory NAVCAM subset.
The dataset is compromised of ~35K images from NASA's Planetary Data System (PDS), including grey-scale navigation camera
(NAVCAM) and color mast camera (Mastcam) from the Curiosity, Opportunity, and Spirit Mars rovers.

| Value | Class |
|---:|---|
| 0 | Soil |
| 1 | Bedrock |
| 2 | Sand |
| 3 | Big Rock |
| 255 | Ignore |

Dataset files are not included in this repository.


## Semantic Baseline

| Component | Configuration |
|---|---|
| Model | DeepLabV3+ |
| Encoder | ResNet-101 |
| Initialization | ImageNet |
| Input | `513 × 513` |
| Output stride | 16 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Batch size | 2 |
| Schedule | Cosine annealing |
| Precision | Automatic mixed precision |
| Epochs | 40 |
| Seed | 42 |

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
