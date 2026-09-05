# AI4Mars onboarding sample

These eight image/mask pairs come from the development-validation split of
AI4Mars merged 0.6, DOI `10.5281/zenodo.15995036`. The dataset is licensed
under CC BY 4.0 by its original authors and contributors.

The sample is for setup validation and visual inspection only. It is not a
benchmark, a test set, or evidence for model selection or scientific claims.
No sealed expert-evaluation image is included.

Selection used only ground-truth metadata. For each of the four NAV classes,
two examples were selected in descending class-fraction order, subject to:

- at least 10% of valid pixels belonging to the target class;
- at least 5% belonging to another NAV class;
- at least 50% of all pixels being labeled; and
- unique source-image and acquisition-sequence identifiers.

Ties are resolved by valid-pixel fraction and then stable source identifier.
Model predictions were never consulted.