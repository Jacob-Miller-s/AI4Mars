"""Prepare an isolated, proposed-protocol clarification review without changing prior evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from src.rock_instance.annotations import (
    PROTOCOL_V2_CALIBRATION_RESOLVED,
    load_review_state,
    save_review_state,
    sha256_file,
    unresolved_candidate_component_ids,
    validate_review_state,
)
from src.rock_instance.calibration_closure import load_agreement_report


CLARIFICATION_SCOPE_NAME = "calibration_clarification"
PROPOSED_PROTOCOL_VERSION = "v2.1-object-identity-clarified-proposed"


def _selection_image_ids(selection_manifest: Path) -> list[str]:
    with Path(selection_manifest).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    image_ids = [row.get("stable_source_image_id", "") for row in rows]
    if not image_ids or any(not image_id for image_id in image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("Clarification selection manifest must contain unique stable_source_image_id values.")
    return image_ids


def initialize_isolated_clarification_state(
    primary_state: dict[str, Any],
    *,
    image_ids: list[str],
    selection_manifest: Path,
    proposed_protocol_path: Path,
    source_repeat_state_sha256: str,
    source_agreement_report_sha256: str,
) -> dict[str, Any]:
    """Make an empty human-review state while retaining immutable source provenance."""
    validate_review_state(primary_state)
    if primary_state.get("review_scope", {}).get("name") != "calibration":
        raise ValueError("Clarification review must be initialized from primary calibration evidence.")
    primary_scope_ids = set(primary_state["review_scope"]["image_ids"])
    if not set(image_ids) <= primary_scope_ids:
        raise ValueError("Clarification selection includes images outside the primary calibration scope.")
    state = copy.deepcopy(primary_state)
    for image in state["images"].values():
        image["annotations"] = []
        image["review_status"] = "unreviewed"
        image["reviewer"] = None
        if "candidate_component_ids" in image:
            image["obvious_candidate_independent_rock_observed"] = False
            image["candidate_independent_observation_note"] = ""
    state["resolution_records"] = []
    state["review_scope"] = {
        "name": CLARIFICATION_SCOPE_NAME,
        "source_manifest": Path(selection_manifest).name,
        "source_manifest_sha256": sha256_file(selection_manifest),
        "image_ids": image_ids,
    }
    state["clarification_review"] = {
        "proposed_protocol_version": PROPOSED_PROTOCOL_VERSION,
        "proposed_protocol_path": str(Path(proposed_protocol_path)),
        "proposed_protocol_sha256": sha256_file(proposed_protocol_path),
        "source_primary_protocol_version": primary_state["protocol"]["version"],
        "source_primary_state_sha256": None,
        "source_repeat_state_sha256": source_repeat_state_sha256,
        "source_agreement_report_sha256": source_agreement_report_sha256,
        "prior_decisions_hidden": True,
    }
    validate_review_state(state)
    return state


def prepare_clarification_review(
    *,
    primary_state_path: Path,
    repeat_state_path: Path,
    agreement_report_path: Path,
    component_candidates_csv: Path,
    selection_manifest: Path,
    proposed_protocol_path: Path,
    output_dir: Path,
) -> Path:
    """Write a six-image empty clarification package; never mutate prior annotations."""
    primary_state_path = Path(primary_state_path)
    repeat_state_path = Path(repeat_state_path)
    agreement_report_path = Path(agreement_report_path)
    component_candidates_csv = Path(component_candidates_csv)
    selection_manifest = Path(selection_manifest)
    proposed_protocol_path = Path(proposed_protocol_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing clarification-review artifacts: {output_dir}")
    if not all(path.is_file() for path in (component_candidates_csv, selection_manifest, proposed_protocol_path)):
        raise FileNotFoundError("Clarification review requires component, selection, and proposed-protocol files.")
    primary_state = load_review_state(primary_state_path)
    repeat_state = load_review_state(repeat_state_path)
    validate_review_state(primary_state)
    validate_review_state(repeat_state)
    if primary_state["protocol"]["version"] != PROTOCOL_V2_CALIBRATION_RESOLVED:
        raise ValueError("Clarification review requires v2.0-calibration-resolved primary provenance.")
    if repeat_state.get("review_scope", {}).get("name") != "calibration_repeat":
        raise ValueError("Clarification review requires a completed isolated repeat state.")
    if any(
        repeat_state["images"][image_id]["review_status"] != "reviewed"
        or unresolved_candidate_component_ids(repeat_state, image_id)
        for image_id in repeat_state["review_scope"]["image_ids"]
    ):
        raise ValueError("Clarification review requires completed repeat coverage.")
    report = load_agreement_report(
        agreement_report_path,
        primary_state_path=primary_state_path,
        repeat_state_path=repeat_state_path,
    )
    if report.get("CALIBRATION_PROTOCOL_RECOMMENDATION") != "CLARIFY":
        raise ValueError("Clarification review is only prepared from a CLARIFY recommendation.")
    expected_component_hash = primary_state["component_review"]["component_manifest_sha256"]
    if sha256_file(component_candidates_csv) != expected_component_hash:
        raise ValueError("Clarification component manifest does not match primary calibration provenance.")
    image_ids = _selection_image_ids(selection_manifest)
    if not set(image_ids) <= set(repeat_state["review_scope"]["image_ids"]):
        raise ValueError("Clarification selection must be a subset of the completed repeat-review scope.")
    output_dir.mkdir(parents=True)
    copied_component_manifest = output_dir / component_candidates_csv.name
    copied_selection_manifest = output_dir / selection_manifest.name
    copied_protocol = output_dir / proposed_protocol_path.name
    shutil.copyfile(component_candidates_csv, copied_component_manifest)
    shutil.copyfile(selection_manifest, copied_selection_manifest)
    shutil.copyfile(proposed_protocol_path, copied_protocol)
    clarification_state = initialize_isolated_clarification_state(
        primary_state,
        image_ids=image_ids,
        selection_manifest=copied_selection_manifest,
        proposed_protocol_path=copied_protocol,
        source_repeat_state_sha256=sha256_file(repeat_state_path),
        source_agreement_report_sha256=sha256_file(agreement_report_path),
    )
    clarification_state["clarification_review"]["source_primary_state_sha256"] = sha256_file(primary_state_path)
    state_path = output_dir / "review_state.json"
    save_review_state(state_path, clarification_state)
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "clarification_review_version": PROPOSED_PROTOCOL_VERSION,
                "primary_state_path": str(primary_state_path),
                "primary_state_sha256": sha256_file(primary_state_path),
                "repeat_state_path": str(repeat_state_path),
                "repeat_state_sha256": sha256_file(repeat_state_path),
                "agreement_report_path": str(agreement_report_path),
                "agreement_report_sha256": sha256_file(agreement_report_path),
                "selection_manifest": copied_selection_manifest.name,
                "selection_manifest_sha256": sha256_file(copied_selection_manifest),
                "proposed_protocol": copied_protocol.name,
                "proposed_protocol_sha256": sha256_file(copied_protocol),
                "component_manifest": copied_component_manifest.name,
                "component_manifest_sha256": sha256_file(copied_component_manifest),
                "prior_decisions_hidden": True,
                "human_decisions_initialized": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return state_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-state-path", required=True, type=Path)
    parser.add_argument("--repeat-state-path", required=True, type=Path)
    parser.add_argument("--agreement-report-path", required=True, type=Path)
    parser.add_argument("--component-candidates-csv", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--proposed-protocol-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        prepare_clarification_review(
            primary_state_path=args.primary_state_path,
            repeat_state_path=args.repeat_state_path,
            agreement_report_path=args.agreement_report_path,
            component_candidates_csv=args.component_candidates_csv,
            selection_manifest=args.selection_manifest,
            proposed_protocol_path=args.proposed_protocol_path,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()