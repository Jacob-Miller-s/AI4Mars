# Semantic Reproduction v1 evidence

This directory is the local evidence bundle prepared for the unpublished `semantic-reproduction-v1` GitHub Release. It does not publish, upload, or duplicate the checkpoint.

## Release asset

- Asset name: `deeplabv3plus-tesla-p100-seed42-best-val-miou.pth`
- Local source: `artifacts/ai4mars-paper-reproduction/frozen/deeplabv3plus-tesla-p100-seed42-best-val-miou.pth`
- Size: 549,040,346 bytes
- SHA-256: `90e74a9071d9bfb180d80ab2bb1927f1ea83a74d7e0601750873c2547a5ddaa3`
- Epoch: 25
- Validation mIoU: `0.8328454614546958`

Before publishing, verify the local source with:

```powershell
Get-FileHash -Algorithm SHA256 artifacts\ai4mars-paper-reproduction\frozen\deeplabv3plus-tesla-p100-seed42-best-val-miou.pth
```

Upload the checkpoint as a binary release asset without modifying it. Publish only after checking that the release tag, asset name, size, and SHA-256 agree with `checkpoint-reference.json`.

## Included evidence

- `checkpoint-reference.json`: artifact identity and publication state
- `checkpoint.sha256`: standard checksum file for the external asset
- `canonical-config.yaml`: newline-normalized copy of the frozen reproduction configuration
- `validation-summary.json`: development-selection result stored in the checkpoint
- `sealed-expert-summary.json`: final sealed results transcribed from the closure record
- `provenance.json`: source revision, environment, dataset, and manifest identities stored in the checkpoint

The checkpoint payload is the source for checkpoint metadata and validation selection. `docs/research/semantic_baseline_closure.md` is the source for validation per-class IoUs and the sealed expert summary because the original evaluation JSON is not present in this checkout.
