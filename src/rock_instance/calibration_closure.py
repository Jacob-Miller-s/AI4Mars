"""Audit completed calibration and prepare its isolated intra-rater repeat review."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from src.rock_instance.annotations import (
    PROTOCOL_V2_CALIBRATION_RESOLVED,
    component_coverage_for_image,
    load_review_state,
    save_review_state,
    sha256_file,
    unresolved_candidate_component_ids,
    validate_review_state,
)
from src.rock_instance.repeat_review import (
    REPEAT_SELECTION_SEED,
    REPEAT_SELECTION_VERSION,
    initialize_isolated_repeat_state,
    select_repeat_image_ids,
)
from src.rock_instance.review_report import summarize_review_state


DEFAULT_REPEAT_TARGET_SIZE = 8
FREEZE_BLOCKED_ACTIONS = (
    "remaining_pilot_review",
    "protocol_freeze",
    "instance_dataset_freeze",
    "mask_rcnn_target_export",
    "model_training",
)


def audit_calibration_closure(
    state: dict[str, Any],
    *,
    repeat_state: dict[str, Any] | None = None,
    agreement_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe completion and report the conditions that still block a protocol freeze."""
    validate_review_state(state)
    if state.get("protocol", {}).get("version") != PROTOCOL_V2_CALIBRATION_RESOLVED:
        raise ValueError("Calibration closure requires a v2.0-calibration-resolved state.")
    scope = state.get("review_scope", {})
    if scope.get("name") != "calibration":
        raise ValueError("Calibration closure requires the primary calibration review scope.")
    image_ids = list(scope["image_ids"])
    images = state["images"]
    incomplete_images = [image_id for image_id in image_ids if images[image_id]["review_status"] != "reviewed"]
    unresolved_by_image = {
        image_id: unresolved_candidate_component_ids(state, image_id)
        for image_id in image_ids
        if unresolved_candidate_component_ids(state, image_id)
    }
    scoped_annotations = [annotation for image_id in image_ids for annotation in images[image_id]["annotations"]]
    status_counts = Counter(annotation["annotation_status"] for annotation in scoped_annotations)
    component_counts = {
        "expected": sum(len(images[image_id]["candidate_component_ids"]) for image_id in image_ids),
        "covered": sum(len(component_coverage_for_image(state, image_id)) for image_id in image_ids),
        "unresolved": sum(len(component_ids) for component_ids in unresolved_by_image.values()),
    }
    primary_complete = not incomplete_images and not unresolved_by_image
    blockers = [] if primary_complete else ["primary_calibration_incomplete"]
    repeat_summary: dict[str, Any] = {"status": "not_initialized"}
    if repeat_state is None:
        blockers.append("isolated_repeat_review_pending")
    else:
        validate_review_state(repeat_state)
        repeat_scope = repeat_state.get("review_scope", {})
        if repeat_scope.get("name") != "calibration_repeat":
            raise ValueError("Closure repeat state must have a calibration_repeat scope.")
        repeat_ids = list(repeat_scope["image_ids"])
        repeat_incomplete = [
            image_id
            for image_id in repeat_ids
            if repeat_state["images"][image_id]["review_status"] != "reviewed"
            or unresolved_candidate_component_ids(repeat_state, image_id)
        ]
        repeat_summary = {
            "status": "complete" if not repeat_incomplete else "pending",
            "images_total": len(repeat_ids),
            "images_incomplete": repeat_incomplete,
            "selection_version": repeat_state.get("repeat_review", {}).get("selection_version"),
            "selection_seed": repeat_state.get("repeat_review", {}).get("selection_seed"),
        }
        if repeat_incomplete:
            blockers.append("isolated_repeat_review_pending")
        else:
            if agreement_report is None:
                blockers.append("repeat_agreement_analysis_pending")
            else:
                recommendation = agreement_report.get("CALIBRATION_PROTOCOL_RECOMMENDATION")
                if agreement_report.get("analysis_type") != "intra-rater consistency" or recommendation not in {"FREEZE", "CLARIFY"}:
                    raise ValueError("Agreement report must be a valid intra-rater consistency analysis with a FREEZE or CLARIFY recommendation.")
                repeat_summary["agreement_analysis"] = {
                    "status": "complete",
                    "recommendation": recommendation,
                    "report_path": agreement_report.get("agreement_report_path"),
                }
                blockers.append(
                    "protocol_clarification_pending"
                    if recommendation == "CLARIFY"
                    else "protocol_freeze_requires_human_approval"
                )
    return {
        "calibration_closure_version": "v1",
        "pilot_id": state["pilot_id"],
        "primary_calibration": {
            "images_total": len(image_ids),
            "images_reviewed": len(image_ids) - len(incomplete_images),
            "images_incomplete": incomplete_images,
            "component_coverage": component_counts,
            "unresolved_component_ids_by_image": unresolved_by_image,
            "terminal_decision_counts": dict(sorted(status_counts.items())),
            "images_with_uncertain_annotations": sum(
                any(annotation["annotation_status"] == "uncertain" for annotation in images[image_id]["annotations"])
                for image_id in image_ids
            ),
            "merge_resolution_records": sum(record["resolution_type"] == "merge" for record in state["resolution_records"]),
            "split_resolution_records": sum(record["resolution_type"] == "split" for record in state["resolution_records"]),
            "complete": primary_complete,
            "review_report": summarize_review_state(state, calibration_image_ids=set(image_ids)),
        },
        "repeat_review": repeat_summary,
        "protocol_freeze_gate": {
            "status": "eligible_for_review" if not blockers else "blocked",
            "blocking_conditions": blockers,
            "prohibited_actions": list(FREEZE_BLOCKED_ACTIONS),
            "statement": "Calibration completion does not authorize pilot expansion, protocol freeze, target export, or model training.",
        },
    }


def prepare_repeat_review(
    *,
    primary_state_path: Path,
    component_candidates_csv: Path,
    output_dir: Path,
    target_size: int = DEFAULT_REPEAT_TARGET_SIZE,
) -> tuple[Path, dict[str, Any]]:
    """Create a deterministic empty repeat state without modifying primary calibration evidence."""
    primary_state_path = Path(primary_state_path)
    component_candidates_csv = Path(component_candidates_csv)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing repeat-review artifacts: {output_dir}")
    state = load_review_state(primary_state_path)
    closure = audit_calibration_closure(state)
    if not closure["primary_calibration"]["complete"]:
        raise ValueError("Repeat review is blocked until primary calibration has no incomplete images or unresolved components.")
    if not component_candidates_csv.is_file():
        raise FileNotFoundError(f"Component candidate manifest is missing: {component_candidates_csv}")
    expected_component_hash = state["component_review"]["component_manifest_sha256"]
    if sha256_file(component_candidates_csv) != expected_component_hash:
        raise ValueError("Component candidate manifest does not match the calibrated review-state provenance.")
    repeat_ids = select_repeat_image_ids(state, target_size=target_size)
    repeat_state = initialize_isolated_repeat_state(state, repeat_ids)
    output_dir.mkdir(parents=True)
    copied_component_manifest = output_dir / component_candidates_csv.name
    shutil.copyfile(component_candidates_csv, copied_component_manifest)
    state_path = output_dir / "review_state.json"
    save_review_state(state_path, repeat_state)
    selection = {
        "selection_version": REPEAT_SELECTION_VERSION,
        "selection_seed": REPEAT_SELECTION_SEED,
        "target_size": target_size,
        "source_primary_state": str(primary_state_path),
        "source_primary_state_sha256": sha256_file(primary_state_path),
        "component_candidate_manifest": copied_component_manifest.name,
        "component_candidate_manifest_sha256": sha256_file(copied_component_manifest),
        "image_ids": repeat_ids,
    }
    (output_dir / "repeat_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return state_path, closure


def load_agreement_report(
    agreement_report_path: Path,
    *,
    primary_state_path: Path,
    repeat_state_path: Path,
) -> dict[str, Any]:
    """Load an intra-rater report only when it is bound to these exact artifacts."""
    agreement_report_path = Path(agreement_report_path)
    report = json.loads(agreement_report_path.read_text(encoding="utf-8"))
    provenance = report.get("provenance", {})
    if report.get("analysis_type") != "intra-rater consistency":
        raise ValueError("Closure requires an intra-rater consistency agreement report.")
    if provenance.get("primary_state_sha256") != sha256_file(primary_state_path):
        raise ValueError("Agreement report primary-state hash does not match the closure artifact.")
    if provenance.get("repeat_state_sha256") != sha256_file(repeat_state_path):
        raise ValueError("Agreement report repeat-state hash does not match the closure artifact.")
    report["agreement_report_path"] = str(agreement_report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-state-path", required=True, type=Path)
    parser.add_argument("--closure-output-json", required=True, type=Path)
    parser.add_argument("--repeat-state-path", type=Path)
    parser.add_argument("--agreement-report-path", type=Path)
    parser.add_argument("--prepare-repeat-output-dir", type=Path)
    parser.add_argument("--component-candidates-csv", type=Path)
    parser.add_argument("--repeat-target-size", type=int, default=DEFAULT_REPEAT_TARGET_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_repeat_output_dir is not None:
        if args.component_candidates_csv is None:
            raise ValueError("--prepare-repeat-output-dir requires --component-candidates-csv.")
        repeat_state_path, _ = prepare_repeat_review(
            primary_state_path=args.primary_state_path,
            component_candidates_csv=args.component_candidates_csv,
            output_dir=args.prepare_repeat_output_dir,
            target_size=args.repeat_target_size,
        )
        repeat_state = load_review_state(repeat_state_path)
    else:
        repeat_state = load_review_state(args.repeat_state_path) if args.repeat_state_path is not None else None
    if args.agreement_report_path is not None and args.repeat_state_path is None and args.prepare_repeat_output_dir is None:
        raise ValueError("--agreement-report-path requires --repeat-state-path.")
    agreement_report = None
    if args.agreement_report_path is not None:
        agreement_repeat_path = repeat_state_path if args.prepare_repeat_output_dir is not None else args.repeat_state_path
        agreement_report = load_agreement_report(
            args.agreement_report_path,
            primary_state_path=args.primary_state_path,
            repeat_state_path=agreement_repeat_path,
        )
    closure = audit_calibration_closure(
        load_review_state(args.primary_state_path), repeat_state=repeat_state, agreement_report=agreement_report,
    )
    args.closure_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.closure_output_json.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(closure, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()