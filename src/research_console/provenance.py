"""Cached inspection of committed manifests and split-isolation evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ManifestProvenanceInspector:
    """Summarize manifest evidence without loading a full dataset into memory."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self.manifest_path = self.repo_root / "artifacts" / "manifests" / "ai4mars_dataset_manifest.csv"
        self.split_dir = self.manifest_path.parent / "splits"
        self._cached_signature: tuple[tuple[str, int, int], ...] | None = None
        self._cached_snapshot: dict[str, Any] | None = None

    def _files(self) -> list[Path]:
        split_files = sorted(self.split_dir.glob("*_nav.csv")) if self.split_dir.is_dir() else []
        return [self.manifest_path, *split_files]

    def _signature(self) -> tuple[tuple[str, int, int], ...]:
        signature = []
        for path in self._files():
            if path.exists():
                stat = path.stat()
                signature.append((path.as_posix(), stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def snapshot(self) -> dict[str, Any]:
        signature = self._signature()
        if self._cached_signature == signature and self._cached_snapshot is not None:
            return self._cached_snapshot

        if not self.manifest_path.exists():
            snapshot = {
                "available": False,
                "gates": [{"name": "manifest_present", "passed": False, "detail": "Manifest file is unavailable."}],
                "issues": ["Dataset manifest is unavailable."],
            }
            self._cached_signature = signature
            self._cached_snapshot = snapshot
            return snapshot

        roles: Counter[str] = Counter()
        schemes: Counter[str] = Counter()
        exclusions: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        total_rows = 0
        included_rows = 0
        first_row: dict[str, str] | None = None
        with self.manifest_path.open("r", newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if first_row is None:
                    first_row = row
                total_rows += 1
                exclusion = (row.get("exclusion_reason") or "").strip()
                if exclusion:
                    exclusions[exclusion] += 1
                    continue
                included_rows += 1
                roles[(row.get("label_role") or "unspecified").strip()] += 1
                schemes[(row.get("label_scheme") or "unspecified").strip()] += 1
                raw_counts = row.get("per_class_pixel_counts_json") or "{}"
                try:
                    for class_id, count in json.loads(raw_counts).items():
                        class_counts[str(class_id)] += int(count)
                except (TypeError, ValueError, json.JSONDecodeError):
                    exclusions["invalid_per_class_pixel_counts_json"] += 1

        split_summaries: dict[str, dict[str, Any]] = {}
        split_source_ids: dict[str, set[str]] = {}
        split_sequence_ids: dict[str, set[str]] = {}
        split_role_errors: list[str] = []
        split_files = sorted(self.split_dir.glob("*_nav.csv")) if self.split_dir.is_dir() else []
        for split_path in split_files:
            with split_path.open("r", newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            split_name = split_path.stem.replace("_nav", "")
            split_source_ids[split_name] = {row.get("stable_source_image_id", "") for row in rows if row.get("stable_source_image_id")}
            split_sequence_ids[split_name] = {row.get("sequence_id", "") for row in rows if row.get("sequence_id")}
            expected_role = "expert_gold_test" if split_name.startswith("test_") else "crowdsourced_train"
            unexpected_roles = sorted({row.get("label_role", "") for row in rows if row.get("label_role", "") != expected_role})
            if unexpected_roles:
                split_role_errors.append(f"{split_name} has unexpected label roles: {', '.join(unexpected_roles)}")
            split_summaries[split_name] = {
                "rows": len(rows),
                "source_groups": len(split_source_ids[split_name]),
                "sequence_groups": len(split_sequence_ids[split_name]),
                "sha256": _sha256_file(split_path),
                "expected_label_role": expected_role,
            }

        overlaps: list[str] = []
        protected_names = [name for name in split_summaries if name.startswith("test_")]
        for left, right in [("train", "val"), *((train_val, test) for train_val in ("train", "val") for test in protected_names)]:
            if left not in split_source_ids or right not in split_source_ids:
                continue
            source_overlap = split_source_ids[left] & split_source_ids[right]
            sequence_overlap = split_sequence_ids[left] & split_sequence_ids[right]
            if source_overlap:
                overlaps.append(f"{left}/{right} source-image overlap: {len(source_overlap)}")
            if sequence_overlap:
                overlaps.append(f"{left}/{right} sequence overlap: {len(sequence_overlap)}")

        gates = [
            {"name": "manifest_present", "passed": True, "detail": "Canonical manifest is present."},
            {
                "name": "required_splits_present",
                "passed": {"train", "val"}.issubset(split_summaries) and bool(protected_names),
                "detail": "Train, crowdsourced validation, and at least one sealed expert-test split are required.",
            },
            {
                "name": "split_roles_match_policy",
                "passed": not split_role_errors,
                "detail": "; ".join(split_role_errors) if split_role_errors else "Split label roles match the crowdsourced/expert policy.",
            },
            {
                "name": "grouped_split_isolation",
                "passed": not overlaps,
                "detail": "; ".join(overlaps) if overlaps else "No source-image or sequence overlap was found.",
            },
        ]
        snapshot = {
            "available": True,
            "dataset_name": "AI4Mars",
            "dataset_version": (first_row or {}).get("dataset_version"),
            "source_record": (first_row or {}).get("dataset_doi"),
            "manifest_path": self.manifest_path.relative_to(self.repo_root).as_posix(),
            "manifest_sha256": _sha256_file(self.manifest_path),
            "pair_counts": {"total_rows": total_rows, "included_rows": included_rows, "excluded_rows": total_rows - included_rows},
            "label_roles": dict(sorted(roles.items())),
            "label_schemes": dict(sorted(schemes.items())),
            "class_pixel_counts": dict(sorted(class_counts.items())),
            "unmatched_or_excluded": dict(sorted(exclusions.items())),
            "grouping_keys": ["stable_source_image_id", "sequence_id"],
            "splits": split_summaries,
            "gates": gates,
            "issues": [gate["detail"] for gate in gates if not gate["passed"]],
        }
        self._cached_signature = signature
        self._cached_snapshot = snapshot
        return snapshot