# AI4Mars Semantic Reproduction

This context defines the scientific language for reproducing the established MSL Curiosity NavCam semantic-segmentation result.

## Language

**Semantic Reproduction MVP**:
The smallest reproducible research path that validates the environment, loads the frozen DeepLabV3+ checkpoint, runs inference on a small attributed AI4Mars sample, and supports inspection of four-class terrain predictions. Its baseline evidence uses fixed MSL Curiosity NavCam manifests and separately sealed expert evaluation.
_Avoid_: Semantic baseline MVP, model demo, training sandbox

**Baseline Evidence**:
The frozen epoch-25 DeepLabV3+ checkpoint, its recorded validation result, and its sealed expert-evaluation artifacts. It is the result future semantic-segmentation research starts from, not a claim that full retraining belongs in first-hour onboarding.
_Avoid_: Best model, production model, ground truth

**Onboarding Sample**:
A small, attributed set of development images and labels used only to validate setup, checkpoint loading, inference, and prediction inspection. It excludes sealed expert examples and is not evaluation evidence.
_Avoid_: Test set, benchmark subset, evaluation sample

**Full Reproduction Run**:
The complete training, validation, resume, and separately invoked sealed expert-evaluation workflow using the fixed reproduction configuration and manifests.
_Avoid_: Onboarding run, smoke test, model demo
