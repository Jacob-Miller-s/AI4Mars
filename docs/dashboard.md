# AI4Mars Research Console

## Purpose and scope

The Research Console is a localhost-only, read-only interface for observing
AI4Mars training records, evaluation evidence, manifest provenance, and safe
artifacts. It is intentionally not a training orchestrator. It does not
start, stop, or modify a training process; safe launch control is deferred.

The console runs alongside an existing training process without importing
Torch or initializing CUDA in its host telemetry sampler. Optional Torch CUDA
allocator memory is supplied only by an already-active training loop.

## Build and launch

Install Python dependencies from the repository root, then install and build
the frontend:

```powershell
pip install -r requirements.txt
cd web
npm install
npm run build
cd ..
```

Launch the production server on loopback:

```powershell
.\.venv\Scripts\python.exe -m src.research_console --host 127.0.0.1 --port 8000
```

Equivalent shell command:

```bash
python -m src.research_console --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The backend serves `web/dist`; rerun `npm run
build` after frontend edits, then restart the Python server after backend
edits.

## Views

- **Overview** shows the active run, protocol-valid best validation run,
  current per-class metrics, failed or invalid records, and system health.
- **Live training** reads durable batch, epoch, and system JSONL events. It
  distinguishes batch loss from epoch train and validation loss, and reconnects
  after a transient event-stream failure.
- **Experiments** searches, filters, sorts, exports, and compares records.
  Cross-manifest, split, or protocol comparisons display warnings rather than
  silently suggesting equivalence.
- **Evaluation** presents raw and row-normalized confusion matrices, per-class
  metrics, and CSV or JSON export. Matrix cells can seed big-rock failure
  filters in the workbench.
- **Workbench** lazily loads indexed images, masks, predictions, overlays, and
  error maps. It supports split filtering, paging, class-mask visibility,
  failure filters, and side-by-side run inspection.
- **Provenance** reads the committed manifest and split evidence, including
  class pixels, label roles, exclusions, hashes, and grouped isolation gates.
- **Artifacts** exposes only declared files below a run's `artifacts/` or
  `checkpoints/` directory, grouped by artifact kind.

## Instrumenting a training loop

`src.train_utils.train_one_epoch` and `src.train_utils.evaluate` remain
backward compatible. Supply a `RunLogger` and epoch metadata to emit bounded
batch events, global count-based evaluation metrics, and host telemetry:

```python
from pathlib import Path

from src.research_console.run_store import RunLogger
from src.train_utils import evaluate, train_one_epoch

logger = RunLogger(Path("outputs/runs"), metadata)
logger.start()
try:
    for epoch in range(1, metadata.training.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch=epoch,
            run_logger=logger,
        )
        metrics = evaluate(
            model,
            validation_loader,
            loss_fn,
            device,
            return_detailed_metrics=True,
            epoch=epoch,
            train_loss=train_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
            run_logger=logger,
        )
    logger.finish()
except BaseException as error:
    logger.fail(error)
    raise
```

Use only a crowdsourced validation split for iterative selection. Expert test
splits must remain sealed until final evaluation. `RunLogger` ranks only
protocol-valid crowdsourced-validation records.

For CUDA training, the instrumentation records allocator `allocated` and
`reserved` bytes only after Torch has already initialized CUDA. The dashboard
process itself never initializes CUDA to obtain those values.

## Synthetic smoke runs

Use the synthetic command only to verify local plumbing:

```powershell
.\.venv\Scripts\python.exe -m src.research_console.demo --runs-root outputs/runs --run-id synthetic-smoke
```

It is CPU-only and writes explicitly invalid-for-benchmark provenance. Never
report its losses, IoU values, or artifacts as AI4Mars findings.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| GPU shows unavailable or no VRAM | `nvidia-ml-py` and a functioning NVML driver are optional. The console remains usable without them. Allocated/reserved values require an active CUDA training loop. |
| An artifact link returns unavailable | The file must be declared by the run and remain below that run's `artifacts/` or `checkpoints/` directory. Absolute paths, traversal, and undeclared locations are rejected. |
| A run is missing or malformed | Inspect `metadata.json`, `summary.json`, and JSONL records. The reader ignores only an incomplete final JSONL line; a malformed completed line intentionally fails closed. |
| Browser cannot reach the API | Confirm the server command is still running on `127.0.0.1:8000`, rebuild `web/dist`, and restart FastAPI after backend edits. Use `VITE_API_BASE` only when deliberately serving the frontend against another local API base. |
| A run cannot become the default best result | It may be legacy, protocol-invalid, sealed-test, incomplete, or belong to an incompatible manifest/split cohort. Review the registry warnings and provenance hashes. |
| A synthetic record appears in the registry | This is expected after a smoke run. It is marked `demo`, `synthetic`, and `non-benchmark`, and is excluded from benchmark ranking. |
| `pytest` is not found | Use `python -m unittest discover -s tests -v`; this repository does not declare `pytest` as a dependency. |

## Validation commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd web
npm test
npm run build
```