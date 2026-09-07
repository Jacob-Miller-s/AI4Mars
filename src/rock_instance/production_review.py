"""Freeze approved calibration protocol v2.3 and prepare its remaining human review scope."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.rock_instance.annotations import (
    APPROVED_PRODUCTION_HASHES,
    BOUNDARY_INDETERMINATE_STATUS,
    CALIBRATION_SIZE,
    CALIBRATION_FINALIZATION_SCHEMA_VERSION,
    PILOT_SIZE,
    PRODUCTION_COMPONENT_COUNT,
    PRODUCTION_REVIEW_PILOT_ID,
    PRODUCTION_REVIEW_PROVENANCE_SCHEMA_VERSION,
    PRODUCTION_REVIEW_SCOPE_NAME,
    PRODUCTION_SIZE,
    PROTOCOL_FREEZE_SCHEMA_VERSION,
    PROTOCOL_FREEZE_STATUS,
    PROTOCOL_V2_3_CALIBRATION_FINAL,
    component_coverage_for_image,
    configure_frozen_production_review,
    configure_review_scope,
    initialize_review_state,
    load_review_state,
    save_review_state,
    sha256_file,
    unresolved_candidate_component_ids,
)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_commit_sha(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _require_hash(path: Path, expected_hash: str, *, description: str) -> None:
    if not Path(path).is_file() or sha256_file(path) != expected_hash:
        raise ValueError(f"{description} does not match its frozen SHA-256 provenance.")


def _closure_markdown(freeze: dict[str, Any], closure: dict[str, Any]) -> str:
    accounting = closure["final_calibration_status_accounting"]
    return "\n".join(
        [
            "# Rock Instance Calibration Closure - v2.3",
            "",
            "## Final Status",
            "",
            "`CALIBRATION_PROTOCOL_STATUS = FROZEN`",
            "",
            f"`CALIBRATION_PROTOCOL_VERSION = {freeze['CALIBRATION_PROTOCOL_VERSION']}`",
            "",
            "## Calibration Record",
            "",
            f"1. The initial 24-image calibration was exploratory and later found insufficiently complete per semantic candidate component.",
            f"2. The corrected component-complete calibration reviewed {accounting['calibration_images']} images and covered {accounting['candidate_components']}/{accounting['candidate_components']} candidate components.",
            "3. An isolated repeat review was conducted as a separate state; agreement evidence was used descriptively and did not declare either pass ground truth.",
            "4. Object-identity clarification separated semantic Big Rock components from physical rock instances.",
            "5. Visible-extent clarification established the full defensible visible image-plane object rule, excluding shadow, hidden geometry, and continuous Bedrock.",
            "6. The final whole-object clarification preserved component 8 as an accepted object but showed no reproducible RGB-supported mask boundary.",
            f"7. That result is recorded as one `boundary_indeterminate` exclusion and remains distinct from the {accounting['uncertain_exclusions']} historical image-level `uncertain` exclusions.",
            "8. v2.3 freezes the final distinction: accepted identity with a reproducible boundary yields a polygon; accepted identity without one excludes the entire image from ordinary Mask R-CNN targets without becoming background.",
            "9. Known limitations: calibration is intentionally difficult-case enriched, uses RGB only, and does not establish physical dimensions, range, stereo/depth, or hazard status.",
            "10. Early calibration acceptance rates are not representative full-dataset statistics and must not be extrapolated as such.",
            "",
            "## Frozen Provenance",
            "",
            f"- Protocol SHA-256: `{freeze['frozen_protocol']['sha256']}`",
            f"- Freeze artifact SHA-256: `{freeze['freeze_artifact_sha256']}`",
            f"- Calibration closure SHA-256: `{freeze['calibration_closure']['sha256']}`",
            f"- Boundary-indeterminate ledger SHA-256: `{freeze['boundary_indeterminate_ledger']['sha256']}`",
            f"- Freeze timestamp (UTC): `{freeze['freeze_timestamp_utc']}`",
            "",
            "The protocol was frozen before expansion to the remaining 126-image production review. This report authorizes neither target export nor model training, and it does not freeze `rock_instance_pilot_v1`.",
            "",
        ]
    )


def freeze_v23_protocol(
    *, protocol_path: Path, calibration_closure_path: Path, boundary_ledger_path: Path,
    repeat_state_path: Path, v21_state_path: Path, v22_state_path: Path, final_state_path: Path,
    final_analysis_path: Path, output_dir: Path, repository_root: Path,
) -> dict[str, Path]:
    """Create the immutable canonical v2.3 protocol freeze from approved evidence."""
    paths = {
        "protocol": Path(protocol_path), "calibration_closure": Path(calibration_closure_path),
        "boundary_indeterminate_ledger": Path(boundary_ledger_path), "repeat_state": Path(repeat_state_path),
        "v21_state": Path(v21_state_path), "v22_state": Path(v22_state_path), "final_state": Path(final_state_path),
        "final_analysis": Path(final_analysis_path),
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("Protocol freeze requires every approved calibration artifact.")
    closure = _load_json(paths["calibration_closure"])
    ledger = _load_json(paths["boundary_indeterminate_ledger"])
    if (
        closure.get("CALIBRATION_PROTOCOL_RECOMMENDATION") != "FREEZE"
        or closure.get("protocol_freeze_gate", {}).get("status") != "eligible_for_human_approval"
    ):
        raise ValueError("Protocol freeze requires the approved v2.3 calibration recommendation.")
    if ledger.get("schema_version") != CALIBRATION_FINALIZATION_SCHEMA_VERSION:
        raise ValueError("Protocol freeze requires the validated boundary-indeterminate ledger.")
    protocol_hash = sha256_file(paths["protocol"])
    if (
        PROTOCOL_V2_3_CALIBRATION_FINAL not in paths["protocol"].read_text(encoding="utf-8")
        or closure.get("protocol", {}).get("version") != PROTOCOL_V2_3_CALIBRATION_FINAL
        or closure["protocol"].get("sha256") != protocol_hash
    ):
        raise ValueError("Protocol freeze requires the approved v2.3 protocol and closure hash match.")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing protocol freeze: {output_dir}")
    output_dir.mkdir(parents=True)
    frozen_protocol_path = output_dir / paths["protocol"].name
    shutil.copyfile(paths["protocol"], frozen_protocol_path)
    freeze = {
        "schema_version": PROTOCOL_FREEZE_SCHEMA_VERSION,
        "CALIBRATION_PROTOCOL_STATUS": PROTOCOL_FREEZE_STATUS,
        "CALIBRATION_PROTOCOL_VERSION": PROTOCOL_V2_3_CALIBRATION_FINAL,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_commit_sha(Path(repository_root)),
        "frozen_protocol": {"path": str(frozen_protocol_path), "sha256": sha256_file(frozen_protocol_path)},
        "calibration_closure": {"path": str(paths["calibration_closure"]), "sha256": sha256_file(paths["calibration_closure"])},
        "boundary_indeterminate_ledger": {"path": str(paths["boundary_indeterminate_ledger"]), "sha256": sha256_file(paths["boundary_indeterminate_ledger"])},
        "calibration_artifacts": {
            label: {"path": str(paths[label]), "sha256": sha256_file(paths[label])}
            for label in ("repeat_state", "v21_state", "v22_state", "final_state", "final_analysis")
        },
        "final_calibration_recommendation": closure["CALIBRATION_PROTOCOL_RECOMMENDATION"],
        "immutability": "Do not modify the frozen protocol in place. Any methodological change requires a new explicit protocol version and calibration review.",
        "prohibited_actions": ["model_training", "mask_rcnn_target_export", "rock_instance_pilot_v1_freeze"],
    }
    freeze_path = output_dir / "protocol_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze["freeze_artifact_sha256"] = sha256_file(freeze_path)
    closure_report_path = output_dir / "calibration_closure_v2.3.md"
    closure_report_path.write_text(_closure_markdown(freeze, closure), encoding="utf-8")
    return {"freeze": freeze_path, "protocol": frozen_protocol_path, "closure_report": closure_report_path}


def summarize_production_review(state: dict[str, Any]) -> dict[str, Any]:
    """Report only production-review progress and target eligibility already supported by human decisions."""
    scope = state.get("review_scope", {})
    if scope.get("name") != PRODUCTION_REVIEW_SCOPE_NAME:
        raise ValueError("Production progress requires the frozen v2.3 production scope.")
    image_ids = list(scope["image_ids"])
    images = state["images"]
    annotations = [annotation for image_id in image_ids for annotation in images[image_id]["annotations"]]
    counts = Counter(annotation["annotation_status"] for annotation in annotations)
    reviewed_ids = [image_id for image_id in image_ids if images[image_id]["review_status"] == "reviewed"]
    unresolved_components = {
        image_id: unresolved_candidate_component_ids(state, image_id)
        for image_id in image_ids
        if unresolved_candidate_component_ids(state, image_id)
    }
    exclusion_reasons: Counter[str] = Counter()
    target_eligible = 0
    for image_id in reviewed_ids:
        statuses = {annotation["annotation_status"] for annotation in images[image_id]["annotations"]}
        if BOUNDARY_INDETERMINATE_STATUS in statuses:
            exclusion_reasons[BOUNDARY_INDETERMINATE_STATUS] += 1
        elif "uncertain" in statuses:
            exclusion_reasons["uncertain"] += 1
        else:
            target_eligible += 1
    resolutions = Counter(record["resolution_type"] for record in state.get("resolution_records", []))
    complete = len(reviewed_ids) == len(image_ids) and not unresolved_components
    return {
        "schema_version": "rock_instance_production_review_progress_v1",
        "protocol_version": state["protocol"]["version"],
        "production_images_total": len(image_ids),
        "production_images_reviewed": len(reviewed_ids),
        "production_images_remaining": len(image_ids) - len(reviewed_ids),
        "reviewed_candidate_components": sum(len(component_coverage_for_image(state, image_id)) for image_id in image_ids),
        "candidate_components_total": sum(len(images[image_id]["candidate_component_ids"]) for image_id in image_ids),
        "candidate_components_unresolved": sum(len(component_ids) for component_ids in unresolved_components.values()),
        "accepted_instances": counts["accepted"],
        "rejected_bedrock": counts["rejected_bedrock"],
        "rejected_noise": counts["rejected_noise"],
        "uncertain": counts["uncertain"],
        "boundary_indeterminate": counts[BOUNDARY_INDETERMINATE_STATUS],
        "resolved_splits": resolutions["split"],
        "resolved_merges": resolutions["merge"],
        "deferred": counts["deferred"] + sum(images[image_id]["review_status"] == "deferred" for image_id in image_ids),
        "unresolved_split_merge": counts["split_required"] + counts["merge_required"],
        "target_eligible_images": target_eligible,
        "target_excluded_images": sum(exclusion_reasons.values()),
        "target_excluded_reasons": dict(sorted(exclusion_reasons.items())),
        "production_review_complete": complete,
        "final_dataset_statistics_ready": False,
        "training_status": "blocked_until_all_150_images_are_reviewed_and_human_approval_is_granted",
    }


def prepare_production_review(
    *, source_pilot_manifest: Path, component_manifest: Path, calibration_manifest: Path,
    frozen_protocol_path: Path, protocol_freeze_path: Path, calibration_state_path: Path,
    boundary_ledger_path: Path, dataset_root: Path, output_dir: Path,
) -> dict[str, Path]:
    """Prepare an empty deterministic 126-image v2.3 production review package."""
    source_pilot_manifest = Path(source_pilot_manifest)
    component_manifest = Path(component_manifest)
    calibration_manifest = Path(calibration_manifest)
    frozen_protocol_path = Path(frozen_protocol_path)
    protocol_freeze_path = Path(protocol_freeze_path)
    calibration_state_path = Path(calibration_state_path)
    boundary_ledger_path = Path(boundary_ledger_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing production review artifacts: {output_dir}")
    if not all(path.is_file() for path in (source_pilot_manifest, component_manifest, calibration_manifest, frozen_protocol_path, protocol_freeze_path, calibration_state_path, boundary_ledger_path)):
        raise FileNotFoundError("Production review requires all frozen protocol and source provenance artifacts.")
    approved_inputs = {
        source_pilot_manifest: "source_pilot_manifest_sha256",
        component_manifest: "component_manifest_sha256",
        calibration_manifest: "calibration_manifest_sha256",
        frozen_protocol_path: "frozen_protocol_sha256",
        protocol_freeze_path: "protocol_freeze_sha256",
        calibration_state_path: "calibration_state_sha256",
        boundary_ledger_path: "boundary_ledger_sha256",
    }
    for path, hash_name in approved_inputs.items():
        _require_hash(
            path,
            APPROVED_PRODUCTION_HASHES[hash_name],
            description=hash_name.removesuffix("_sha256").replace("_", " ").title(),
        )
    freeze = _load_json(protocol_freeze_path)
    if freeze.get("schema_version") != PROTOCOL_FREEZE_SCHEMA_VERSION or freeze.get("CALIBRATION_PROTOCOL_STATUS") != PROTOCOL_FREEZE_STATUS or freeze.get("CALIBRATION_PROTOCOL_VERSION") != PROTOCOL_V2_3_CALIBRATION_FINAL:
        raise ValueError("Production review requires the canonical frozen v2.3 protocol artifact.")
    _require_hash(frozen_protocol_path, freeze["frozen_protocol"]["sha256"], description="Frozen protocol")
    _require_hash(boundary_ledger_path, freeze["boundary_indeterminate_ledger"]["sha256"], description="Boundary-indeterminate ledger")
    pilot_rows, pilot_fields = _read_csv(source_pilot_manifest)
    calibration_rows, _ = _read_csv(calibration_manifest)
    pilot_ids = [row["stable_source_image_id"] for row in pilot_rows]
    calibration_ids = [row["stable_source_image_id"] for row in calibration_rows]
    if len(pilot_rows) != PILOT_SIZE or len(set(pilot_ids)) != PILOT_SIZE:
        raise ValueError("Production review requires the original deterministic 150-image pilot manifest.")
    if len(calibration_ids) != CALIBRATION_SIZE or len(set(calibration_ids)) != CALIBRATION_SIZE or not set(calibration_ids) <= set(pilot_ids):
        raise ValueError("Production review requires the completed 24-image calibration manifest within the original pilot.")
    remaining_rows = [row for row in pilot_rows if row["stable_source_image_id"] not in set(calibration_ids)]
    if len(remaining_rows) != PRODUCTION_SIZE:
        raise ValueError("Production review scope is not exactly the remaining 126 pilot images.")
    remaining_ids = [row["stable_source_image_id"] for row in remaining_rows]
    component_rows, _ = _read_csv(component_manifest)
    component_ids_by_image: dict[str, list[int]] = {row["stable_source_image_id"]: [] for row in remaining_rows}
    for component in component_rows:
        image_id = component["stable_source_image_id"]
        if image_id in component_ids_by_image:
            component_ids_by_image[image_id].append(int(component["component_id"]))
    if any(not component_ids for component_ids in component_ids_by_image.values()):
        raise ValueError("Every remaining production image must retain candidate-component provenance.")
    if sum(len(component_ids) for component_ids in component_ids_by_image.values()) != PRODUCTION_COMPONENT_COUNT:
        raise ValueError("Production review scope must contain exactly 814 approved candidate components.")
    output_dir.mkdir(parents=True)
    copied_pilot_path = output_dir / source_pilot_manifest.name
    copied_component_path = output_dir / component_manifest.name
    copied_calibration_path = output_dir / calibration_manifest.name
    for source, destination in ((source_pilot_manifest, copied_pilot_path), (component_manifest, copied_component_path), (calibration_manifest, copied_calibration_path)):
        shutil.copyfile(source, destination)
    remaining_manifest_path = output_dir / "rock_instance_pilot_remaining_126_v2.3.csv"
    _write_csv(remaining_manifest_path, remaining_rows, pilot_fields)
    _require_hash(
        remaining_manifest_path,
        APPROVED_PRODUCTION_HASHES["remaining_scope_manifest_sha256"],
        description="Remaining scope manifest",
    )
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": PRODUCTION_REVIEW_PROVENANCE_SCHEMA_VERSION,
                "protocol_freeze_sha256": sha256_file(protocol_freeze_path),
                "frozen_protocol_sha256": sha256_file(frozen_protocol_path),
                "calibration_state_sha256": sha256_file(calibration_state_path),
                "boundary_ledger_sha256": sha256_file(boundary_ledger_path),
                "source_pilot_manifest_sha256": sha256_file(source_pilot_manifest),
                "calibration_manifest_sha256": sha256_file(calibration_manifest),
                "component_manifest_sha256": sha256_file(component_manifest),
                "remaining_scope_images": len(remaining_ids),
                "expert_splits_excluded": True,
                "historical_calibration_annotations_immutable": True,
                "production_review_status": "not_started",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    closure_path = Path(
        str(freeze["calibration_closure"]["path"]).replace("\\", "/")
    )
    if not closure_path.is_absolute():
        closure_path = Path(__file__).resolve().parents[2] / closure_path
    state = initialize_review_state(copied_pilot_path, Path(dataset_root), pilot_id=PRODUCTION_REVIEW_PILOT_ID)
    configure_review_scope(state, name=PRODUCTION_REVIEW_SCOPE_NAME, image_ids=remaining_ids, source_manifest=remaining_manifest_path)
    configure_frozen_production_review(
        state,
        component_manifest=copied_component_path,
        component_ids_by_image=component_ids_by_image,
        frozen_protocol_path=frozen_protocol_path,
        protocol_freeze_path=protocol_freeze_path,
        calibration_state_path=calibration_state_path,
        source_pilot_manifest=copied_pilot_path,
        calibration_manifest=copied_calibration_path,
        boundary_ledger_path=boundary_ledger_path,
        provenance_path=provenance_path,
        calibration_closure_path=closure_path,
        remaining_scope_manifest_path=remaining_manifest_path,
    )
    state_path = output_dir / "review_state.json"
    save_review_state(state_path, state)
    progress_path = output_dir / "review_progress.json"
    progress_path.write_text(json.dumps(summarize_production_review(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"state": state_path, "remaining_manifest": remaining_manifest_path, "progress": progress_path, "provenance": provenance_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    for name in ("protocol_path", "calibration_closure_path", "boundary_ledger_path", "repeat_state_path", "v21_state_path", "v22_state_path", "final_state_path", "final_analysis_path", "output_dir", "repository_root"):
        freeze_parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    prepare_parser = subparsers.add_parser("prepare")
    for name in ("source_pilot_manifest", "component_manifest", "calibration_manifest", "frozen_protocol_path", "protocol_freeze_path", "calibration_state_path", "boundary_ledger_path", "dataset_root", "output_dir"):
        prepare_parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("--state-path", required=True, type=Path)
    progress_parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_v23_protocol(**{key: value for key, value in vars(args).items() if key != "command"})
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))
    elif args.command == "prepare":
        result = prepare_production_review(**{key: value for key, value in vars(args).items() if key != "command"})
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))
    else:
        report = summarize_production_review(load_review_state(args.state_path))
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()