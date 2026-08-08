"""Prepare a human-approval-only v2.3 calibration conclusion from immutable evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from src.rock_instance.annotations import (
    BOUNDARY_INDETERMINATE_STATUS,
    annotation_component_ids,
    load_review_state,
    sha256_file,
    unresolved_candidate_component_ids,
    validate_boundary_indeterminate_record,
)
from src.rock_instance.boundary_review import (
    BOUNDARY_REVIEW_SCHEMA_VERSION,
    FINAL_CLARIFICATION_SCHEMA_VERSION,
    FINAL_CLARIFICATION_TARGET_ID,
    load_boundary_review_state,
)


FINAL_PROTOCOL_VERSION = "v2.3-calibration-final"
FINALIZATION_SCHEMA_VERSION = "rock_instance_calibration_finalization_v1"
FINALIZATION_TARGET_ID = FINAL_CLARIFICATION_TARGET_ID


def _completed_target(state: dict[str, Any], *, target_id: str, require_single_scope: bool) -> dict[str, Any]:
    scope_ids = state.get("review_scope", {}).get("target_ids", [])
    if target_id not in scope_ids or require_single_scope and scope_ids != [target_id]:
        raise ValueError("Finalization received an invalid component-8 review scope.")
    targets = [target for target in state.get("targets", []) if target.get("target_id") == target_id]
    if len(targets) != 1:
        raise ValueError("Finalization requires exactly one component-8 target record.")
    target = targets[0]
    if target.get("review_status") != "redrawn" or target.get("object_identity_fixed") != "accepted" or target.get("identity_escalation"):
        raise ValueError("Finalization requires a completed accepted component-8 review without escalation.")
    return target


def _load_final_analysis(path: Path, *, expected_hashes: dict[str, str]) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("analysis_type") != "v2.2.1 final whole-object clarification consistency":
        raise ValueError("Finalization requires the completed v2.2.1 one-object consistency analysis.")
    provenance = report.get("provenance", {})
    if any(provenance.get(label) != digest for label, digest in expected_hashes.items()):
        raise ValueError("Final consistency analysis provenance does not match immutable review artifacts.")
    target = report.get("target", {})
    if target.get("target_id") != FINALIZATION_TARGET_ID or target.get("source_candidate_component_id") != 8:
        raise ValueError("Final consistency analysis does not describe component 8.")
    if report.get("CALIBRATION_PROTOCOL_RECOMMENDATION") != "CLARIFY_AGAIN":
        raise ValueError("Finalization expects the documented unresolved-boundary analysis outcome.")
    return report


def prepare_calibration_finalization(
    *, primary_state_path: Path, repeat_state_path: Path, v21_state_path: Path, v22_state_path: Path,
    final_state_path: Path, final_analysis_path: Path, final_protocol_path: Path, output_dir: Path,
) -> dict[str, Path]:
    """Write a separate v2.3 finalization ledger and approval-gated closure report."""
    paths = {
        "primary": Path(primary_state_path), "repeat": Path(repeat_state_path), "v2.1": Path(v21_state_path),
        "v2.2": Path(v22_state_path), "v2.2.1": Path(final_state_path),
    }
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise ValueError("Finalization requires five distinct immutable review artifacts.")
    before_hashes = {label: sha256_file(path) for label, path in paths.items()}
    primary, repeat, v21 = (load_review_state(paths[label]) for label in ("primary", "repeat", "v2.1"))
    v22 = load_boundary_review_state(paths["v2.2"])
    final = load_boundary_review_state(paths["v2.2.1"])
    if primary.get("review_scope", {}).get("name") != "calibration" or any(primary["images"][image_id]["review_status"] != "reviewed" for image_id in primary["review_scope"]["image_ids"]):
        raise ValueError("Finalization requires a completed primary calibration state.")
    if repeat.get("review_scope", {}).get("name") != "calibration_repeat" or any(repeat["images"][image_id]["review_status"] != "reviewed" for image_id in repeat["review_scope"]["image_ids"]):
        raise ValueError("Finalization requires a completed isolated repeat state.")
    if v21.get("review_scope", {}).get("name") != "calibration_clarification":
        raise ValueError("Finalization requires the completed v2.1 clarification state.")
    if v22.get("schema_version") != BOUNDARY_REVIEW_SCHEMA_VERSION or final.get("schema_version") != FINAL_CLARIFICATION_SCHEMA_VERSION:
        raise ValueError("Finalization requires the completed v2.2 and v2.2.1 review artifacts.")
    v22_target = _completed_target(v22, target_id=FINALIZATION_TARGET_ID, require_single_scope=False)
    final_target = _completed_target(final, target_id=FINALIZATION_TARGET_ID, require_single_scope=True)
    expected_final_provenance = {
        "primary_state_sha256": before_hashes["primary"], "repeat_state_sha256": before_hashes["repeat"],
        "v21_state_sha256": before_hashes["v2.1"], "source_boundary_state_sha256": before_hashes["v2.2"],
    }
    if any(final["provenance"].get(key) != digest for key, digest in expected_final_provenance.items()):
        raise ValueError("Final v2.2.1 state provenance does not match immutable review artifacts.")
    if v22_target["image_id"] != final_target["image_id"] or v22_target["source_candidate_component_id"] != final_target["source_candidate_component_id"]:
        raise ValueError("Final v2.2.1 state does not preserve the component-8 source target.")
    expected_analysis_provenance = dict(before_hashes)
    final_analysis_path = Path(final_analysis_path)
    analysis = _load_final_analysis(final_analysis_path, expected_hashes=expected_analysis_provenance)
    final_protocol_path = Path(final_protocol_path)
    if not final_protocol_path.is_file() or FINAL_PROTOCOL_VERSION not in final_protocol_path.read_text(encoding="utf-8"):
        raise ValueError("Finalization requires the v2.3 calibration-final protocol.")
    v21_annotation = next(
        (
            annotation for annotation in v21["images"][final_target["image_id"]]["annotations"]
            if annotation["instance_id"] == final_target["v21_instance_id"]
            and annotation["annotation_status"] == "accepted"
            and final_target["source_candidate_component_id"] in annotation_component_ids(annotation)
        ),
        None,
    )
    if v21_annotation is None:
        raise ValueError("Finalization cannot preserve accepted identity provenance for component 8.")
    primary_image = primary["images"][final_target["image_id"]]
    record = {
        "record_id": f"{final_target['image_id']}:component-{final_target['source_candidate_component_id']}:boundary-indeterminate-v1",
        "image_id": final_target["image_id"],
        "sequence_id": final_target["sequence_id"],
        "source_candidate_component_ids": [final_target["source_candidate_component_id"]],
        "object_identity": "accepted",
        "boundary_status": "indeterminate",
        "accepted_identity_reference": v21_annotation["instance_id"],
        "final_clarification_target_id": final_target["target_id"],
        "final_clarification_state_sha256": before_hashes["v2.2.1"],
        "final_analysis_sha256": sha256_file(final_analysis_path),
        "reason_evidence": "Multiple independent hidden-prior redraws retained accepted object identity but produced materially incompatible visible extents under explicit whole-object rules; RGB evidence is insufficient for reproducible instance-mask ground truth.",
    }
    record = validate_boundary_indeterminate_record(primary, record)
    scope_ids = list(primary["review_scope"]["image_ids"])
    uncertain_images = {
        image_id for image_id in scope_ids
        if any(annotation["annotation_status"] == "uncertain" for annotation in primary["images"][image_id]["annotations"])
    }
    boundary_images = {record["image_id"]}
    protocol_hash = sha256_file(final_protocol_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing v2.3 finalization artifacts: {output_dir}")
    output_dir.mkdir(parents=True)
    copied_protocol = output_dir / final_protocol_path.name
    shutil.copyfile(final_protocol_path, copied_protocol)
    provenance = {**before_hashes, "final_analysis_sha256": sha256_file(final_analysis_path), "final_protocol_sha256": sha256_file(copied_protocol)}
    ledger = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "protocol": {"version": FINAL_PROTOCOL_VERSION, "path": str(copied_protocol), "sha256": protocol_hash},
        "provenance": provenance,
        "boundary_indeterminate_records": [record],
        "ordinary_maskrcnn_target_policy": {
            "boundary_indeterminate_status": BOUNDARY_INDETERMINATE_STATUS,
            "exclude_whole_image": True,
            "positive_mask_target": False,
            "implicit_background": False,
            "ignore_region_mechanism": "not introduced",
        },
    }
    closure = {
        "calibration_closure_version": "v2.3",
        "protocol": ledger["protocol"],
        "provenance": provenance,
        "final_calibration_status_accounting": {
            "calibration_images": len(scope_ids),
            "candidate_components": sum(len(primary["images"][image_id]["candidate_component_ids"]) for image_id in scope_ids),
            "unresolved_candidate_components": sum(len(unresolved_candidate_component_ids(primary, image_id)) for image_id in scope_ids),
            "boundary_indeterminate_exclusions": len(boundary_images),
            "uncertain_exclusions": len(uncertain_images),
            "ordinary_target_ineligible_images": len(boundary_images | uncertain_images),
            "boundary_indeterminate_record_ids": [record["record_id"]],
        },
        "CALIBRATION_PROTOCOL_RECOMMENDATION": "FREEZE",
        "protocol_freeze_gate": {
            "status": "eligible_for_human_approval",
            "blocking_conditions": ["protocol_freeze_requires_human_approval"],
            "prohibited_actions": ["remaining_pilot_review", "protocol_freeze", "instance_dataset_freeze", "mask_rcnn_target_export", "model_training"],
            "statement": "The v2.3 recommendation resolves the final calibration contradiction by excluding a real but boundary-indeterminate object from ordinary targets. Human approval is still required before protocol activation or any remaining-image review.",
        },
        "remaining_review_workflow": {
            "status": "not_started",
            "precondition": "Explicit human approval of the v2.3 protocol-freeze recommendation.",
            "command": "python -m src.rock_instance.review_tool --interactive --state-path <approved_v2.3_remaining_review_state> --component-candidates-csv <approved_component_manifest> --dataset-root data/raw/ai4mars/ai4mars-dataset-merged-0.6 --reviewer single_researcher",
            "requirement": "Create the approved versioned remaining-review state before this command; do not reuse or modify immutable calibration artifacts.",
        },
    }
    ledger_path = output_dir / "boundary_indeterminate_exclusions.json"
    closure_path = output_dir / "calibration_closure_v2.3.json"
    provenance_path = output_dir / "provenance.json"
    for path, payload in ((ledger_path, ledger), (closure_path, closure), (provenance_path, provenance)):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if before_hashes != {label: sha256_file(path) for label, path in paths.items()}:
        raise RuntimeError("Finalization unexpectedly modified an immutable review artifact.")
    return {"ledger": ledger_path, "closure": closure_path, "provenance": provenance_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-state-path", required=True, type=Path)
    parser.add_argument("--repeat-state-path", required=True, type=Path)
    parser.add_argument("--v21-state-path", required=True, type=Path)
    parser.add_argument("--v22-state-path", required=True, type=Path)
    parser.add_argument("--final-state-path", required=True, type=Path)
    parser.add_argument("--final-analysis-path", required=True, type=Path)
    parser.add_argument("--final-protocol-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({key: str(value) for key, value in prepare_calibration_finalization(**vars(args)).items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()