"""Compare a completed v2.1 clarification pass with immutable primary and repeat evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.rock_instance.annotations import (
    annotation_component_ids,
    load_review_state,
    sha256_file,
    unresolved_candidate_component_ids,
    validate_review_state,
)
from src.rock_instance.intra_rater_consistency import (
    DIRECT_DISPOSITIONS,
    SOURCE_FIELDS,
    _accepted_objects,
    _canonical_digest,
    _structure_summary,
    component_dispositions,
    mask_iou,
    match_accepted_objects,
)


ANALYSIS_VERSION = "clarification-comparison-v1"
V21_RULES_BY_IMAGE = {
    "NLB_463551084EDR_F0411534NCAM00385M1": {
        "original_discrepancy": "13-versus-5 subdivision of a continuous layered component plus a rock/noise switch.",
        "v21_rule": "A layered band or texture variation is not a split; continuous outcrop is rejected_bedrock and only a visibly bounded isolated surface is accepted.",
    },
    "NLB_517255503EDR_F0541610NCAM07753M1": {
        "original_discrepancy": "Different child counts within components 1 and 2 plus secondary-feature disposition switches.",
        "v21_rule": "Split only at a terrain gap, overlap/occlusion edge, or stable separative contour; otherwise use uncertain when no outer identity is defensible.",
    },
    "NLB_483955685EDR_F0470598NCAM00320M1": {
        "original_discrepancy": "Bedrock-versus-accepted switch for component 3 and incompatible component-1 terminal history.",
        "v21_rule": "Accept only a visibly bounded discrete surface; use rejected_bedrock for continuous terrain and uncertain when the visible identity remains indeterminate.",
    },
    "NLB_490004046EDR_F0482122NCAM00281M1": {
        "original_discrepancy": "Multiple incompatible subdivisions of a broad component and a secondary-object switch.",
        "v21_rule": "A broad continuous shelf is rejected_bedrock; thin semantic fragments do not establish separate object identity.",
    },
    "NLB_548252623EDR_F0631150NCAM00312M1": {
        "original_discrepancy": "Component 4 changed from noise to an accepted secondary object.",
        "v21_rule": "A small feature is accepted only with a complete terrain-separated visible contour; otherwise it is noise or uncertain.",
    },
    "NLB_528261206EDR_F0580738NCAM00385M1": {
        "original_discrepancy": "Primary merged components 3 and 4 while repeat rejected 3 and accepted 4 directly.",
        "v21_rule": "Merge only when the visible surface is continuous across contributors; adjacency and semantic connectivity are insufficient.",
    },
}
V21_BOUNDARY_RECHECK_BY_IMAGE = {
    "NLB_463551084EDR_F0411534NCAM00385M1": {
        "status": "needs_clarification",
        "rationale": "The retained component-4 object has primary-to-v2.1 IoU 0.0263, while the historic primary/repeat component-4 match was 0.6894; the v2.1 visible-extent rule did not reproduce a stable boundary.",
    },
    "NLB_517255503EDR_F0541610NCAM07753M1": {
        "status": "not_applicable",
        "rationale": "v2.1 records no accepted object: the rule resolves the prior child-count dispute as terminal uncertainty rather than a polygon-boundary comparison.",
    },
    "NLB_483955685EDR_F0470598NCAM00320M1": {
        "status": "needs_clarification",
        "rationale": "The retained component-8 object matches v2.1 at IoU 0.1702 against primary and 0.2619 against repeat; the visible extent remains materially divergent.",
    },
    "NLB_490004046EDR_F0482122NCAM00281M1": {
        "status": "not_applicable",
        "rationale": "v2.1 records no accepted object: the broad shelf is rejected_bedrock and fragments are noise.",
    },
    "NLB_548252623EDR_F0631150NCAM00312M1": {
        "status": "needs_clarification",
        "rationale": "Component 3 remains accepted but has IoU 0.0008 against primary and 0.0025 against repeat, even though component 8 is comparatively stable; the boundary rule needs a clearer operational treatment for this retained object.",
    },
    "NLB_528261206EDR_F0580738NCAM00385M1": {
        "status": "consistent",
        "rationale": "All three retained v2.1 objects have nontrivial one-to-one overlap with both historic passes, and the component-3/4 merge ambiguity is resolved without an inferred merge.",
    },
}


def _accepted_count(image: dict[str, Any]) -> int:
    return sum(annotation["annotation_status"] == "accepted" for annotation in image["annotations"])


def _status_counts(image: dict[str, Any]) -> dict[str, int]:
    return {status: sum(annotation["annotation_status"] == status for annotation in image["annotations"]) for status in DIRECT_DISPOSITIONS}


def _object_geometry_rows(
    historical_image: dict[str, Any], clarification_image: dict[str, Any], *, image_id: str, historical_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matches, unmatched_historical, unmatched_clarification = match_accepted_objects(historical_image, clarification_image)
    match_rows = [
        {
            "image_id": image_id,
            "comparison": f"{historical_label}_to_v2.1",
            "historical_instance_id": match["primary_instance_id"],
            "v21_instance_id": match["repeat_instance_id"],
            "historical_source_candidate_component_ids": match["primary_source_candidate_component_ids"],
            "v21_source_candidate_component_ids": match["repeat_source_candidate_component_ids"],
            "mask_iou": match["mask_iou"],
            "historical_area_pixels": match["primary_area_pixels"],
            "v21_area_pixels": match["repeat_area_pixels"],
        }
        for match in matches
    ]
    clarification_objects = _accepted_objects(clarification_image)
    historical_objects = _accepted_objects(historical_image)
    unmatched_rows = []
    for side, objects, other_objects in (
        (historical_label, unmatched_historical, clarification_objects),
        ("v2.1", unmatched_clarification, historical_objects),
    ):
        for item in objects:
            unmatched_rows.append(
                {
                    "image_id": image_id,
                    "comparison": f"{historical_label}_to_v2.1",
                    "side": side,
                    "instance_id": item["annotation"]["instance_id"],
                    "source_candidate_component_ids": annotation_component_ids(item["annotation"]),
                    "best_iou_against_opposite": max(
                        (mask_iou(item["mask"], other["mask"]) for other in other_objects), default=0.0,
                    ),
                }
            )
    return match_rows, unmatched_rows


def verify_clarification_provenance(
    primary_state_path: Path, repeat_state_path: Path, clarification_state_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fail closed unless the completed v2.1 scope is bound to both historic artifacts."""
    primary_state_path = Path(primary_state_path)
    repeat_state_path = Path(repeat_state_path)
    clarification_state_path = Path(clarification_state_path)
    if len({primary_state_path.resolve(), repeat_state_path.resolve(), clarification_state_path.resolve()}) != 3:
        raise ValueError("Primary, repeat, and clarification artifacts must be distinct files.")
    primary = load_review_state(primary_state_path)
    repeat = load_review_state(repeat_state_path)
    clarification = load_review_state(clarification_state_path)
    validate_review_state(primary)
    validate_review_state(repeat)
    validate_review_state(clarification)
    clarification_ids = list(clarification.get("review_scope", {}).get("image_ids", []))
    if primary.get("review_scope", {}).get("name") != "calibration":
        raise ValueError("Clarification comparison requires primary calibration scope.")
    if repeat.get("review_scope", {}).get("name") != "calibration_repeat":
        raise ValueError("Clarification comparison requires isolated repeat scope.")
    if clarification.get("review_scope", {}).get("name") != "calibration_clarification" or len(clarification_ids) != 6:
        raise ValueError("Clarification comparison requires exactly six calibration_clarification images.")
    if set(clarification_ids) != set(V21_RULES_BY_IMAGE):
        raise ValueError("Clarification scope does not match the declared six-image v2.1 evidence set.")
    if not set(clarification_ids) <= set(repeat["review_scope"]["image_ids"]):
        raise ValueError("Clarification images must be a subset of the isolated repeat scope.")
    primary_sha256 = sha256_file(primary_state_path)
    repeat_sha256 = sha256_file(repeat_state_path)
    metadata = clarification.get("clarification_review", {})
    if metadata.get("source_primary_state_sha256") != primary_sha256:
        raise ValueError("Clarification source primary hash does not match the supplied artifact.")
    if metadata.get("source_repeat_state_sha256") != repeat_sha256:
        raise ValueError("Clarification source repeat hash does not match the supplied artifact.")
    if not metadata.get("prior_decisions_hidden"):
        raise ValueError("Clarification state does not attest that prior decisions were hidden.")
    protocol_path = Path(metadata.get("proposed_protocol_path", ""))
    if not protocol_path.is_file() or metadata.get("proposed_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("Clarification proposed-protocol provenance is invalid.")
    if primary["component_review"]["component_manifest_sha256"] != repeat["component_review"]["component_manifest_sha256"] != clarification["component_review"]["component_manifest_sha256"]:
        raise ValueError("Review states do not share component-manifest provenance.")
    source_mismatch = [
        image_id
        for image_id in clarification_ids
        if any(
            primary["images"][image_id][field] != repeat["images"][image_id][field]
            or primary["images"][image_id][field] != clarification["images"][image_id][field]
            for field in SOURCE_FIELDS
        )
    ]
    if source_mismatch:
        raise ValueError(f"Clarification source-image provenance differs: {source_mismatch}")
    incomplete = [
        image_id
        for image_id in clarification_ids
        if any(
            state["images"][image_id]["review_status"] != "reviewed"
            or unresolved_candidate_component_ids(state, image_id)
            for state in (primary, repeat, clarification)
        )
    ]
    if incomplete:
        raise ValueError(f"Clarification comparison requires complete coverage: {incomplete}")
    return primary, repeat, clarification, {
        "primary_state_path": str(primary_state_path),
        "primary_state_sha256": primary_sha256,
        "repeat_state_path": str(repeat_state_path),
        "repeat_state_sha256": repeat_sha256,
        "clarification_state_path": str(clarification_state_path),
        "clarification_state_sha256": sha256_file(clarification_state_path),
        "clarification_scope_size": len(clarification_ids),
        "image_ids": clarification_ids,
        "proposed_protocol_version": metadata["proposed_protocol_version"],
        "proposed_protocol_sha256": metadata["proposed_protocol_sha256"],
        "prior_decisions_hidden": True,
        "artifacts_distinct": True,
    }


def recommendation_for_resolution_rows(rows: list[dict[str, Any]]) -> str:
    """Recommend protocol freeze consideration only when every targeted discrepancy is resolved."""
    return "FREEZE" if rows and all(row["fully_resolved_by_v21_rule"] for row in rows) else "CLARIFY_AGAIN"


def analyze_clarification_pass(
    primary_state_path: Path, repeat_state_path: Path, clarification_state_path: Path,
) -> dict[str, Any]:
    """Produce a non-mutating three-state comparison without selecting a historic ground truth."""
    paths = (Path(primary_state_path), Path(repeat_state_path), Path(clarification_state_path))
    before_hashes = {str(path): sha256_file(path) for path in paths}
    primary, repeat, clarification, provenance = verify_clarification_provenance(*paths)
    before_digests = {"primary": _canonical_digest(primary), "repeat": _canonical_digest(repeat), "v2.1": _canonical_digest(clarification)}
    per_image = []
    component_rows = []
    geometry_rows = []
    unmatched_rows = []
    for image_id in provenance["image_ids"]:
        primary_image = primary["images"][image_id]
        repeat_image = repeat["images"][image_id]
        v21_image = clarification["images"][image_id]
        dispositions = {
            "primary": component_dispositions(primary, image_id),
            "repeat": component_dispositions(repeat, image_id),
            "v2.1": component_dispositions(clarification, image_id),
        }
        for component_id in v21_image["candidate_component_ids"]:
            component_rows.append(
                {
                    "image_id": image_id,
                    "component_id": component_id,
                    **{
                        f"{label}_{field}": disposition.get(field)
                        for label, disposition in dispositions.items()
                        for field in ("kind", "disposition", "reason")
                    },
                }
            )
        primary_matches, primary_unmatched = _object_geometry_rows(primary_image, v21_image, image_id=image_id, historical_label="primary")
        repeat_matches, repeat_unmatched = _object_geometry_rows(repeat_image, v21_image, image_id=image_id, historical_label="repeat")
        geometry_rows.extend(primary_matches + repeat_matches)
        unmatched_rows.extend(primary_unmatched + repeat_unmatched)
        v21_has_structured_component = any(item["kind"] != "direct" for item in dispositions["v2.1"].values())
        v21_rule = V21_RULES_BY_IMAGE[image_id]
        boundary_recheck = V21_BOUNDARY_RECHECK_BY_IMAGE[image_id]
        ontology_resolved = not v21_has_structured_component
        per_image.append(
            {
                "image_id": image_id,
                "primary_accepted_instances": _accepted_count(primary_image),
                "repeat_accepted_instances": _accepted_count(repeat_image),
                "v21_accepted_instances": _accepted_count(v21_image),
                "primary_terminal_status_counts": _status_counts(primary_image),
                "repeat_terminal_status_counts": _status_counts(repeat_image),
                "v21_terminal_status_counts": _status_counts(v21_image),
                "primary_structure": _structure_summary(primary, image_id),
                "repeat_structure": _structure_summary(repeat, image_id),
                "v21_structure": _structure_summary(clarification, image_id),
                "primary_to_v21_structure_changed": _structure_summary(primary, image_id) != _structure_summary(clarification, image_id),
                "repeat_to_v21_structure_changed": _structure_summary(repeat, image_id) != _structure_summary(clarification, image_id),
                "v21_has_structured_component": v21_has_structured_component,
                "original_discrepancy": v21_rule["original_discrepancy"],
                "v21_rule_applied": v21_rule["v21_rule"],
                "ontology_resolved_by_v21_rule": ontology_resolved,
                "boundary_recheck_status": boundary_recheck["status"],
                "boundary_recheck_rationale": boundary_recheck["rationale"],
                "fully_resolved_by_v21_rule": ontology_resolved and boundary_recheck["status"] != "needs_clarification",
                "resolution_basis": "Every v2.1 candidate has one terminal direct disposition; accepted instances are limited to visibly bounded surfaces, and unresolved visual identity is terminal uncertain rather than an arbitrary split or merge.",
            }
        )
    recommendation = recommendation_for_resolution_rows(per_image)
    ious_by_comparison = defaultdict(list)
    for row in geometry_rows:
        ious_by_comparison[row["comparison"]].append(row["mask_iou"])
    geometry_summary = {
        comparison: {
            "matched_objects": len(ious),
            "mean_mask_iou": mean(ious) if ious else None,
            "median_mask_iou": median(ious) if ious else None,
            "min_mask_iou": min(ious, default=None),
            "max_mask_iou": max(ious, default=None),
        }
        for comparison, ious in sorted(ious_by_comparison.items())
    }
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_type": "v2.1 clarification comparison",
        "provenance": provenance,
        "per_image": per_image,
        "component_transitions": component_rows,
        "accepted_object_geometry": {
            "matching_method": "Within-image deterministic maximum mask-IoU one-to-one Hungarian assignment; zero-IoU assignments are unmatched.",
            "summary_by_historical_comparison": geometry_summary,
            "per_object": geometry_rows,
            "unmatched_objects": unmatched_rows,
        },
        "resolution_summary": {
            "targeted_discrepancies": len(per_image),
            "ontology_resolved_by_v21_rule": sum(row["ontology_resolved_by_v21_rule"] for row in per_image),
            "fully_resolved_by_v21_rule": sum(row["fully_resolved_by_v21_rule"] for row in per_image),
            "unresolved_image_ids": [row["image_id"] for row in per_image if not row["fully_resolved_by_v21_rule"]],
            "interpretation": "The clarification pass is compared symmetrically with both historical passes; agreement with either historic pass is evidence of transition, not correctness.",
        },
        "CALIBRATION_PROTOCOL_RECOMMENDATION": recommendation,
        "limitations": [
            "This is a six-image targeted clarification recheck, not inter-rater reliability, expert agreement, model evaluation, or a dataset freeze.",
            "The v2.1 pass is a fresh adjudication; primary and repeat are immutable evidence and neither is designated ground truth.",
            "Object identity, terminal uncertainty, and split/merge structure are evaluated before mask IoU; no universal IoU threshold is used.",
            "A FREEZE recommendation concerns only whether the written v2.1 rules admit coherent targeted interpretations; it does not activate, freeze, export, train, or expand review scope.",
        ],
    }
    after_digests = {"primary": _canonical_digest(primary), "repeat": _canonical_digest(repeat), "v2.1": _canonical_digest(clarification)}
    if before_digests != after_digests:
        raise RuntimeError("Clarification analysis unexpectedly mutated a state in memory.")
    if before_hashes != {str(path): sha256_file(path) for path in paths}:
        raise RuntimeError("Clarification analysis unexpectedly modified an annotation artifact.")
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for field, value in row.items()})


def write_analysis_outputs(report: dict[str, Any], output_dir: Path, markdown_path: Path) -> None:
    """Write stable machine-readable evidence and a concise research-facing conclusion."""
    output_dir = Path(output_dir)
    markdown_path = Path(markdown_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clarification_comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "per_image.csv", report["per_image"])
    _write_csv(output_dir / "component_transitions.csv", report["component_transitions"])
    _write_csv(output_dir / "matched_instances.csv", report["accepted_object_geometry"]["per_object"])
    _write_csv(output_dir / "unmatched_instances.csv", report["accepted_object_geometry"]["unmatched_objects"])
    rows = report["per_image"]
    table = ["| Image | Primary | Repeat | v2.1 | Ontology resolved | Boundary recheck | Full resolution |", "| --- | ---: | ---: | ---: | --- | --- | --- |"]
    table.extend(
        f"| {row['image_id']} | {row['primary_accepted_instances']} | {row['repeat_accepted_instances']} | {row['v21_accepted_instances']} | {row['ontology_resolved_by_v21_rule']} | {row['boundary_recheck_status']} | {row['fully_resolved_by_v21_rule']} |"
        for row in rows
    )
    transitions = []
    for row in rows:
        transitions.extend(
            [
                f"### {row['image_id']}",
                "",
                f"- Original discrepancy: {row['original_discrepancy']}",
                f"- v2.1 rule: {row['v21_rule_applied']}",
                f"- Ontology outcome: {'resolved' if row['ontology_resolved_by_v21_rule'] else 'requires further clarification'}.",
                f"- Boundary recheck: {row['boundary_recheck_status']}. {row['boundary_recheck_rationale']}",
                f"- Full outcome: {'resolved' if row['fully_resolved_by_v21_rule'] else 'requires further clarification'}.",
                "",
                *[
                    f"- {label}: {row[f'{label}_terminal_status_counts']}"
                    for label in ("primary", "repeat", "v21")
                ],
                "",
            ]
        )
    geometry = report["accepted_object_geometry"]["summary_by_historical_comparison"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                "# v2.1 Clarification Comparison",
                "",
                "## Scope",
                "",
                "This analysis compares the completed isolated six-image v2.1 clarification pass symmetrically against immutable primary and repeat evidence. Neither historical pass is treated as ground truth.",
                "",
                "## Provenance",
                "",
                f"- Primary SHA-256: `{report['provenance']['primary_state_sha256']}`",
                f"- Repeat SHA-256: `{report['provenance']['repeat_state_sha256']}`",
                f"- Clarification SHA-256: `{report['provenance']['clarification_state_sha256']}`",
                f"- Proposed protocol: `{report['provenance']['proposed_protocol_version']}` (`{report['provenance']['proposed_protocol_sha256']}`)",
                "",
                "## Instance Counts",
                "",
                *table,
                "",
                "## Object Geometry",
                "",
                *[
                    f"- {comparison}: {summary['matched_objects']} matched objects; mean IoU {summary['mean_mask_iou']:.4f}, median {summary['median_mask_iou']:.4f}."
                    for comparison, summary in geometry.items()
                    if summary["matched_objects"]
                ],
                f"- Unmatched accepted objects across both historical comparisons: {len(report['accepted_object_geometry']['unmatched_objects'])}",
                "",
                "## Per-Image Resolution",
                "",
                *transitions,
                "## Recommendation",
                "",
                f"`CALIBRATION_PROTOCOL_RECOMMENDATION = {report['CALIBRATION_PROTOCOL_RECOMMENDATION']}`",
                "",
                *[f"- {item}" for item in report["limitations"]],
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-state-path", required=True, type=Path)
    parser.add_argument("--repeat-state-path", required=True, type=Path)
    parser.add_argument("--clarification-state-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--markdown-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_clarification_pass(
        args.primary_state_path, args.repeat_state_path, args.clarification_state_path,
    )
    write_analysis_outputs(report, args.output_dir, args.markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()