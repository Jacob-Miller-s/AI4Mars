# ADR 0001: Local-first research console

## Status

Accepted

## Context

AI4Mars training currently runs from notebooks and reusable Python helpers. The
project already records manifests, split hashes, checkpoint metadata, and
evaluation artifacts, but it has no live, queryable experiment record.
The console must run beside training on an 11 GB GTX 1080 Ti without reserving
CUDA memory or requiring a cloud service.

## Decision

The console uses a FastAPI backend bound to `127.0.0.1` and a React/TypeScript
frontend built with Vite. Run directories are the source of truth:

```
outputs/runs/<run_id>/
  metadata.json
  metrics.jsonl
  system_metrics.jsonl
  summary.json
  artifacts/
  checkpoints/
```

Records use a versioned Pydantic schema. JSON snapshots are atomically
replaced; JSONL events are append-only, flushed, bounded in frequency, and
read defensively after interrupted writes. SQLite is intentionally deferred:
the backend can rebuild an index from portable records when needed. GPU
telemetry uses NVML only when available and never initializes CUDA.

The backend serves the production frontend build. Development keeps FastAPI
and Vite as separate processes. APIs expose only run roots and validated
relative artifact paths; they do not execute arbitrary commands or bind beyond
localhost by default.

## Consequences

- Existing notebooks remain usable while `src.train_utils` gains optional
  instrumentation hooks.
- Validation and sealed expert-test records are explicitly distinct, and
  protocol-invalid or legacy runs are excluded from default rankings.
- Large checkpoints and generated telemetry remain ignored by Git; summaries
  are small, portable review artifacts.
- Browser views can reconnect by rereading durable events rather than relying
  on in-memory training state.