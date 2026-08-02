# Research Console Run Schema

## Durable record layout

Each run is a portable directory under `outputs/runs/<run_id>/`:

```text
outputs/runs/<run_id>/
  metadata.json
  metrics.jsonl
  system_metrics.jsonl
  summary.json
  artifacts/
  checkpoints/
```

The schema version is currently `1`. `metadata.json` and `summary.json` are
written atomically with a temporary file and replace. JSONL events are
append-only, flushed, and `fsync`ed. Readers ignore an incomplete final JSONL
line, but reject malformed completed lines.

## Metadata

`RunMetadata` identifies the experiment and must include:

- A portable `run_id`, experiment name, optional hypothesis, tags, and notes.
- Dataset manifest SHA-256, split hashes, split role, and protocol gates.
- Git commit/branch/dirty state and random seeds when available.
- Model architecture, encoder, input resolution, optimizer, loss, batch size,
  learning rate, epochs, and precision mode.
- Runtime environment such as Python, PyTorch, CUDA, GPU, CPU, and memory.
- Declared `ArtifactRef` values with relative POSIX paths only.

The console uses provenance as a ranking guard. Only completed,
protocol-valid, crowdsourced-validation runs are candidates for a default
best-run selection. Incompatible manifest/split cohorts never share that
ranking silently.

## Event streams

`metrics.jsonl` contains two event types:

- `batch`: epoch, batch position, loss, optional smoothed loss, throughput,
  and ETA. These should be bounded by the logger's batch interval.
- `epoch`: train and validation loss, pixel accuracy, global mIoU, raw
  confusion counts, per-class support/IoU/Dice/precision/recall, learning
  rate, epoch duration, and optional checkpoint reference.

`system_metrics.jsonl` contains CPU, RAM, disk I/O, optional NVML GPU
utilization/temperature/VRAM, and optional training-process CUDA allocator
allocated/reserved bytes. The standalone sampler does not import Torch or
initialize CUDA.

## Summary and terminal states

`summary.json` is written when a run reaches a terminal state:

- `completed`
- `failed`
- `interrupted`
- `invalid`

It includes the terminal timestamps, best eligible validation epoch and mIoU,
protocol record, failure reason when applicable, and declared artifacts. A
failure stores its traceback as the portable artifact
`artifacts/failure_traceback.txt`; it does not expose a local traceback path.

## Artifact and sample index safety

Artifacts must be declared with a relative POSIX path and are served only from
the selected run's `artifacts/` or `checkpoints/` directory. Absolute paths,
Windows drive paths, backslashes, dot components, and traversal are rejected.

Optional workbench records live at `artifacts/prediction_index.jsonl`. Each
object can include:

```json
{
  "sample_id": "stable-image-id",
  "split": "crowdsourced_validation",
  "image_iou": 0.64,
  "loss": 0.82,
  "uncertainty": 0.17,
  "big_rock_false_negative": true,
  "big_rock_to_soil": false,
  "assets": {
    "image": "artifacts/samples/image.png",
    "ground_truth": "artifacts/samples/ground_truth.png",
    "prediction": "artifacts/samples/prediction.png",
    "overlay": "artifacts/samples/overlay.png",
    "error_heatmap": "artifacts/samples/error_heatmap.png"
  }
}
```

The workbench orders `image_iou` from lowest to highest, and `loss` or
`uncertainty` from highest to lowest. It can filter by declared split and
big-rock failure signals.

## Compatibility and migration

The server can adapt selected historical artifacts beneath `artifacts/runs/`
as `legacy` records for inspection. Legacy records are visibly labeled and are
not eligible for default benchmark ranking. New schema versions should retain
this portable record design, document migration behavior, and add contract
tests before changing reader semantics.