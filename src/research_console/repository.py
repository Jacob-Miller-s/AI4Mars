"""Read-only query layer over portable AI4Mars experiment records."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from .provenance import ManifestProvenanceInspector
from .run_store import RUN_ID_PATTERN, RunReader, iter_run_directories
from .schema import RunStatus, SplitRole


class RunNotFoundError(FileNotFoundError):
    """Raised when a requested run id is not present in the configured root."""


class UnsafeArtifactPathError(ValueError):
    """Raised when an artifact request would leave its permitted directory."""


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def _last_epoch_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((event for event in reversed(events) if event.get("event_type") == "epoch"), None)


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, nested in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(nested, nested_prefix))
        return flattened
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


class RunRepository:
    """Resolve and summarize dashboard records rooted under ``outputs/runs``."""

    def __init__(self, repo_root: Path, runs_root: Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.runs_root = (Path(runs_root) if runs_root is not None else self.repo_root / "outputs" / "runs").resolve()
        self.provenance_inspector = ManifestProvenanceInspector(self.repo_root)

    def _legacy_records(self) -> list[dict[str, Any]]:
        """Adapt old lightweight run summaries without treating them as benchmarks."""
        legacy_root = self.repo_root / "artifacts" / "runs"
        if not legacy_root.is_dir():
            return []
        records = []
        for run_dir in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "metrics.json"
            if not config_path.exists() or not metrics_path.exists():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                split_hashes_path = run_dir / "split_manifest_hashes.json"
                split_hashes = json.loads(split_hashes_path.read_text(encoding="utf-8")) if split_hashes_path.exists() else {}
                manifest_path = run_dir / "dataset_manifest_hash.txt"
                manifest_hash = manifest_path.read_text(encoding="utf-8").strip() if manifest_path.exists() else ""
            except (OSError, json.JSONDecodeError):
                continue
            run_id = f"legacy-{run_dir.name}"
            event = {
                "event_type": "epoch",
                "epoch": metrics.get("epoch"),
                "train_loss": metrics.get("train_loss"),
                "val_loss": metrics.get("val_loss"),
                "pixel_accuracy": metrics.get("pixel_acc"),
                "mean_iou": metrics.get("mean_iou"),
                "per_class_iou": metrics.get("per_class_iou"),
            }
            metadata = {
                "schema_version": 1,
                "run_id": run_id,
                "experiment_name": run_dir.name,
                "tags": ["legacy", "historical"],
                "status": "completed",
                "legacy": True,
                "provenance": {
                    "dataset_name": "AI4Mars",
                    "dataset_version": "unknown",
                    "dataset_manifest_sha256": manifest_hash,
                    "split_manifest_hashes": split_hashes,
                    "split_role": "crowdsourced_validation",
                    "protocol": {
                        "valid": False,
                        "failed_gates": ["legacy_artifact_not_reproduced_under_current_protocol"],
                        "notes": ["Historical artifact; excluded from default ranking."],
                    },
                },
                "model": {"name": config.get("model_name", "unknown")},
                "training": {
                    "optimizer": "unknown",
                    "loss": config.get("loss_name", "unknown"),
                    "learning_rate": config.get("learning_rate"),
                    "batch_size": config.get("batch_size"),
                    "epochs": config.get("epochs"),
                    "class_weights": config.get("loss_weights"),
                },
                "artifact_refs": [],
            }
            summary = {
                "schema_version": 1,
                "run_id": run_id,
                "status": "completed",
                "best_epoch": metrics.get("epoch"),
                "best_validation_mean_iou": metrics.get("mean_iou"),
                "protocol": metadata["provenance"]["protocol"],
            }
            records.append({"metadata": metadata, "summary": summary, "metrics": [event], "system_metrics": []})
        return records

    def _legacy_detail(self, run_id: str) -> dict[str, Any] | None:
        return next((record for record in self._legacy_records() if record["metadata"]["run_id"] == run_id), None)

    def _reader_for_id(self, run_id: str) -> RunReader:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise RunNotFoundError(run_id)
        candidate = (self.runs_root / run_id).resolve()
        if candidate.parent != self.runs_root or not (candidate / "metadata.json").is_file():
            raise RunNotFoundError(run_id)
        return RunReader(candidate)

    def readers(self) -> list[RunReader]:
        return [RunReader(path) for path in iter_run_directories(self.runs_root)]

    def run_card(self, reader: RunReader) -> dict[str, Any]:
        metadata = reader.metadata()
        summary = reader.summary()
        latest_epoch = _last_epoch_event(reader.metrics())
        summary_payload = _json_value(summary) if summary else None
        return {
            "run_id": metadata.run_id,
            "experiment_name": metadata.experiment_name,
            "hypothesis": metadata.hypothesis,
            "researcher_notes": metadata.researcher_notes,
            "tags": metadata.tags,
            "status": metadata.status.value,
            "started_at": metadata.started_at,
            "ended_at": metadata.ended_at,
            "protocol_valid": metadata.provenance.protocol.valid,
            "failed_gates": metadata.provenance.protocol.failed_gates,
            "split_role": metadata.provenance.split_role.value,
            "dataset_version": metadata.provenance.dataset_version,
            "dataset_manifest_sha256": metadata.provenance.dataset_manifest_sha256,
            "split_manifest_hashes": metadata.provenance.split_manifest_hashes,
            "git_commit": metadata.provenance.git_commit,
            "git_branch": metadata.provenance.git_branch,
            "random_seeds": metadata.provenance.random_seeds,
            "model": metadata.model.name,
            "encoder": metadata.model.encoder,
            "latest_epoch": latest_epoch,
            "summary": summary_payload,
        }

    def list_runs(
        self,
        *,
        status: RunStatus | None = None,
        protocol_valid: bool | None = None,
        query: str | None = None,
        manifest_hash: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        cards: list[dict[str, Any]] = []
        warnings: list[str] = []
        for reader in self.readers():
            try:
                card = self.run_card(reader)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                warnings.append(f"Malformed run {reader.run_dir.name}: {error}")
                continue
            if status is not None and card["status"] != status.value:
                continue
            if protocol_valid is not None and card["protocol_valid"] != protocol_valid:
                continue
            if manifest_hash is not None and card["dataset_manifest_sha256"] != manifest_hash:
                continue
            if query:
                haystack = " ".join(
                    [
                        card["run_id"],
                        card["experiment_name"],
                        card["hypothesis"] or "",
                        " ".join(card["tags"]),
                    ]
                ).lower()
                if query.lower() not in haystack:
                    continue
            cards.append(card)
        for legacy in self._legacy_records():
            metadata = legacy["metadata"]
            event = legacy["metrics"][0]
            card = {
                "run_id": metadata["run_id"],
                "experiment_name": metadata["experiment_name"],
                "hypothesis": None,
                "researcher_notes": None,
                "tags": metadata["tags"],
                "status": metadata["status"],
                "started_at": None,
                "ended_at": None,
                "protocol_valid": False,
                "failed_gates": metadata["provenance"]["protocol"]["failed_gates"],
                "split_role": metadata["provenance"]["split_role"],
                "dataset_version": metadata["provenance"]["dataset_version"],
                "dataset_manifest_sha256": metadata["provenance"]["dataset_manifest_sha256"],
                "split_manifest_hashes": metadata["provenance"]["split_manifest_hashes"],
                "git_commit": None,
                "git_branch": None,
                "random_seeds": {},
                "model": metadata["model"]["name"],
                "encoder": None,
                "latest_epoch": event,
                "summary": legacy["summary"],
                "legacy": True,
            }
            if status is not None and card["status"] != status.value:
                continue
            if protocol_valid is not None and protocol_valid:
                continue
            if manifest_hash is not None and card["dataset_manifest_sha256"] != manifest_hash:
                continue
            if query and query.lower() not in " ".join([card["run_id"], card["experiment_name"], *card["tags"]]).lower():
                continue
            cards.append(card)
        cards.sort(
            key=lambda card: (
                card["started_at"].isoformat()
                if hasattr(card["started_at"], "isoformat")
                else card["started_at"] or "",
                card["run_id"],
            ),
            reverse=True,
        )
        return {"total": len(cards), "offset": offset, "limit": limit, "runs": cards[offset : offset + limit], "warnings": warnings}

    def detail(self, run_id: str) -> dict[str, Any]:
        legacy = self._legacy_detail(run_id)
        if legacy is not None:
            return legacy
        reader = self._reader_for_id(run_id)
        return {
            "metadata": _json_value(reader.metadata()),
            "summary": _json_value(reader.summary()) if reader.summary() else None,
            "metrics": reader.metrics(),
            "system_metrics": reader.system_metrics(),
        }

    def events(self, run_id: str, *, after: int = 0) -> dict[str, Any]:
        if after < 0:
            raise ValueError("after must be non-negative.")
        reader = self._reader_for_id(run_id)
        events = [
            *({"stream": "metrics", "event": event} for event in reader.metrics()),
            *({"stream": "system", "event": event} for event in reader.system_metrics()),
        ]
        events.sort(key=lambda item: item["event"].get("timestamp", ""))
        return {"next": len(events), "events": events[after:]}

    def overview(self) -> dict[str, Any]:
        cards = self.list_runs(limit=10000)["runs"]
        active = next((card for card in cards if card["status"] == RunStatus.RUNNING.value), None)
        valid_validation = [
            card
            for card in cards
            if card["protocol_valid"]
            and card["split_role"] == SplitRole.CROWDSOURCED_VALIDATION.value
            and card["status"] == RunStatus.COMPLETED.value
            and card["summary"]
            and card["summary"].get("best_validation_mean_iou") is not None
        ]
        cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for card in valid_validation:
            cohort_key = (
                card["dataset_manifest_sha256"],
                json.dumps(card["split_manifest_hashes"], sort_keys=True),
            )
            cohorts.setdefault(cohort_key, []).append(card)
        provenance = self.provenance()
        current_manifest = provenance.get("manifest_sha256")
        current_splits = {
            split_name: split["sha256"]
            for split_name, split in provenance.get("splits", {}).items()
        }
        current_cohort_key = (current_manifest, json.dumps(current_splits, sort_keys=True))
        ranking_warning = None
        if current_cohort_key in cohorts:
            eligible_for_default = cohorts[current_cohort_key]
        elif len(cohorts) <= 1:
            eligible_for_default = next(iter(cohorts.values()), [])
        else:
            eligible_for_default = []
            ranking_warning = "Multiple incompatible manifest/split cohorts exist; no cross-cohort best run is selected."
        best = max(
            eligible_for_default,
            key=lambda card: card["summary"]["best_validation_mean_iou"],
            default=None,
        )
        failed = [card for card in cards if card["status"] in {RunStatus.FAILED.value, RunStatus.INVALID.value}]
        health = None
        if active is not None:
            active_detail = self.detail(active["run_id"])
            system_events = active_detail["system_metrics"]
            health = system_events[-1] if system_events else None
        return {
            "active_run": active,
            "best_protocol_valid_validation_run": best,
            "recent_runs": cards[:10],
            "failed_runs": failed[:10],
            "system_health": health,
            "expert_test_locked": True,
            "ranking_warning": ranking_warning,
            "provenance_cohorts": [
                {
                    "dataset_manifest_sha256": manifest_hash,
                    "split_manifest_hashes": json.loads(split_hashes),
                    "run_count": len(cohort_runs),
                    "best_run": max(
                        cohort_runs,
                        key=lambda card: card["summary"]["best_validation_mean_iou"],
                    ),
                }
                for (manifest_hash, split_hashes), cohort_runs in cohorts.items()
            ],
        }

    def provenance(self) -> dict[str, Any]:
        return self.provenance_inspector.snapshot()

    def compare(self, run_ids: list[str]) -> dict[str, Any]:
        if len(run_ids) < 2:
            raise ValueError("Select at least two runs for comparison.")
        details = [self.detail(run_id) for run_id in run_ids]
        metadata = [detail["metadata"] for detail in details]
        manifest_hashes = {item["provenance"]["dataset_manifest_sha256"] for item in metadata}
        split_hashes = {
            json.dumps(item["provenance"]["split_manifest_hashes"], sort_keys=True)
            for item in metadata
        }
        split_roles = {item["provenance"]["split_role"] for item in metadata}
        warnings: list[str] = []
        if len(manifest_hashes) > 1:
            warnings.append("Selected runs use different dataset manifest hashes.")
        if len(split_hashes) > 1:
            warnings.append("Selected runs use different split manifest hashes.")
        if len(split_roles) > 1:
            warnings.append("Selected runs use different split roles and are not directly comparable.")
        if any(not item["provenance"]["protocol"]["valid"] for item in metadata):
            warnings.append("At least one selected run failed protocol gates and is excluded from valid ranking.")

        flattened = [_flatten_config(item) for item in metadata]
        all_keys = sorted(set().union(*(item.keys() for item in flattened)))
        config_diff = {
            key: [item.get(key) for item in flattened]
            for key in all_keys
            if len({json.dumps(item.get(key), sort_keys=True, default=str) for item in flattened}) > 1
        }
        return {"runs": details, "warnings": warnings, "config_diff": config_diff}

    def artifact_path(self, run_id: str, artifact_path: str) -> Path:
        reader = self._reader_for_id(run_id)
        if not artifact_path or "\\" in artifact_path:
            raise UnsafeArtifactPathError(artifact_path)
        relative = PurePosixPath(artifact_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise UnsafeArtifactPathError(artifact_path)
        if not relative.parts or relative.parts[0] not in {"artifacts", "checkpoints"}:
            raise UnsafeArtifactPathError(artifact_path)
        candidate = (reader.run_dir / Path(*relative.parts)).resolve()
        if reader.run_dir not in candidate.parents or not candidate.is_file():
            raise UnsafeArtifactPathError(artifact_path)
        return candidate

    def samples(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 50,
        sort_by: str = "image_iou",
        split: str | None = None,
        big_rock_false_negative: bool = False,
        big_rock_to_soil: bool = False,
    ) -> dict[str, Any]:
        reader = self._reader_for_id(run_id)
        index_path = reader.run_dir / "artifacts" / "prediction_index.jsonl"
        if not index_path.exists():
            return {"total": 0, "offset": offset, "limit": limit, "samples": [], "available": False}
        from .run_store import read_jsonl_tolerant

        rows = read_jsonl_tolerant(index_path)
        if big_rock_false_negative:
            rows = [row for row in rows if row.get("big_rock_false_negative")]
        if big_rock_to_soil:
            rows = [row for row in rows if row.get("big_rock_to_soil")]
        available_splits = sorted({str(row["split"]) for row in rows if row.get("split")})
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]
        reverse = sort_by in {"loss", "uncertainty"}
        missing_value = float("-inf") if reverse else float("inf")
        rows.sort(
            key=lambda row: row.get(sort_by) if row.get(sort_by) is not None else missing_value,
            reverse=reverse,
        )
        return {
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "samples": rows[offset : offset + limit],
            "available": True,
            "available_splits": available_splits,
        }