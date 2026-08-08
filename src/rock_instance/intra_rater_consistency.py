"""Compare completed primary and isolated repeat calibration annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from src.rock_instance.annotations import (
    TERMINAL_ANNOTATION_STATUSES,
    annotation_component_ids,
    load_review_state,
    polygon_to_mask,
    sha256_file,
    unresolved_candidate_component_ids,
    validate_review_state,
)


DIRECT_DISPOSITIONS = ("accepted", "rejected_noise", "rejected_bedrock", "uncertain")
SOURCE_FIELDS = ("image_path", "mask_path", "image_width", "image_height", "sequence_id", "candidate_component_ids")


def _canonical_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _resolution_components(state: dict[str, Any], image_id: str) -> set[int]:
    return {
        component_id
        for record in state["resolution_records"]
        if record["image_id"] == image_id
        for component_id in record["source_candidate_component_ids"]
    }


def component_dispositions(state: dict[str, Any], image_id: str) -> dict[int, dict[str, str]]:
    """Return direct terminal dispositions while preserving structural exclusions."""
    image = state["images"][image_id]
    decisions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in image["annotations"]:
        if annotation["annotation_status"] in TERMINAL_ANNOTATION_STATUSES:
            for component_id in annotation_component_ids(annotation):
                decisions[component_id].append(annotation)
    resolved_components = _resolution_components(state, image_id)
    result: dict[int, dict[str, str]] = {}
    for component_id in image["candidate_component_ids"]:
        annotations = decisions[component_id]
        if component_id in resolved_components:
            result[component_id] = {"kind": "structured", "reason": "explicit_resolution"}
        elif len(annotations) != 1 or annotation_component_ids(annotations[0]) != [component_id]:
            result[component_id] = {"kind": "structured", "reason": "multi_annotation_or_multi_component"}
        else:
            result[component_id] = {"kind": "direct", "disposition": annotations[0]["annotation_status"]}
    return result


def mask_iou(primary_mask: torch.Tensor, repeat_mask: torch.Tensor) -> float:
    """Compute foreground mask IoU for two same-shaped boolean masks."""
    if primary_mask.shape != repeat_mask.shape:
        raise ValueError("Mask IoU requires masks with identical geometry.")
    union = int((primary_mask | repeat_mask).sum().item())
    if union == 0:
        return 0.0
    return int((primary_mask & repeat_mask).sum().item()) / union


def maximum_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Solve a deterministic rectangular maximum-weight assignment in $O(n^3)$."""
    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("Assignment weights must be rectangular.")
    transposed = row_count > column_count
    matrix = weights if not transposed else [list(row) for row in zip(*weights)]
    rows, columns = len(matrix), len(matrix[0])
    if rows > columns:
        raise ValueError("Internal assignment matrix must have at least as many columns as rows.")

    # Hungarian minimum-cost form. Sorted inputs and first-index tie handling make ties reproducible.
    potentials_rows = [0.0] * (rows + 1)
    potentials_columns = [0.0] * (columns + 1)
    matched_column_for_row = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row_index in range(1, rows + 1):
        matched_column_for_row[0] = row_index
        column_zero = 0
        minimum_slack = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column_zero] = True
            current_row = matched_column_for_row[column_zero]
            delta = float("inf")
            next_column = 0
            for column_index in range(1, columns + 1):
                if used[column_index]:
                    continue
                cost = -matrix[current_row - 1][column_index - 1]
                reduced_cost = cost - potentials_rows[current_row] - potentials_columns[column_index]
                if reduced_cost < minimum_slack[column_index]:
                    minimum_slack[column_index] = reduced_cost
                    predecessor[column_index] = column_zero
                if minimum_slack[column_index] < delta:
                    delta = minimum_slack[column_index]
                    next_column = column_index
            for column_index in range(columns + 1):
                if used[column_index]:
                    potentials_rows[matched_column_for_row[column_index]] += delta
                    potentials_columns[column_index] -= delta
                else:
                    minimum_slack[column_index] -= delta
            column_zero = next_column
            if matched_column_for_row[column_zero] == 0:
                break
        while True:
            previous_column = predecessor[column_zero]
            matched_column_for_row[column_zero] = matched_column_for_row[previous_column]
            column_zero = previous_column
            if column_zero == 0:
                break
    pairs = [(row_index - 1, column_index - 1) for column_index, row_index in enumerate(matched_column_for_row) if column_index and row_index]
    mapped_pairs = [(column_index, row_index) for row_index, column_index in pairs] if transposed else pairs
    return sorted(mapped_pairs)


def _accepted_objects(image: dict[str, Any]) -> list[dict[str, Any]]:
    objects = []
    for annotation in sorted(image["annotations"], key=lambda item: item["instance_id"]):
        if annotation["annotation_status"] != "accepted":
            continue
        mask = polygon_to_mask(annotation["polygon"], image_width=image["image_width"], image_height=image["image_height"])
        objects.append({"annotation": annotation, "mask": mask, "area_pixels": int(mask.sum().item())})
    return objects


def match_accepted_objects(primary_image: dict[str, Any], repeat_image: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Match accepted instances by maximum polygon-mask IoU within one image only."""
    primary_objects = _accepted_objects(primary_image)
    repeat_objects = _accepted_objects(repeat_image)
    weights = [[mask_iou(primary["mask"], repeat["mask"]) for repeat in repeat_objects] for primary in primary_objects]
    pairs = maximum_weight_assignment(weights)
    matched_primary = set()
    matched_repeat = set()
    matches = []
    for primary_index, repeat_index in pairs:
        iou = weights[primary_index][repeat_index]
        if iou == 0:
            continue
        primary = primary_objects[primary_index]
        repeat = repeat_objects[repeat_index]
        matched_primary.add(primary_index)
        matched_repeat.add(repeat_index)
        matches.append(
            {
                "primary_instance_id": primary["annotation"]["instance_id"],
                "repeat_instance_id": repeat["annotation"]["instance_id"],
                "primary_source_candidate_component_ids": annotation_component_ids(primary["annotation"]),
                "repeat_source_candidate_component_ids": annotation_component_ids(repeat["annotation"]),
                "mask_iou": iou,
                "primary_area_pixels": primary["area_pixels"],
                "repeat_area_pixels": repeat["area_pixels"],
                "repeat_to_primary_area_ratio": repeat["area_pixels"] / primary["area_pixels"],
                "absolute_area_difference_pixels": abs(repeat["area_pixels"] - primary["area_pixels"]),
            }
        )
    unmatched_primary = [primary_objects[index] for index in range(len(primary_objects)) if index not in matched_primary]
    unmatched_repeat = [repeat_objects[index] for index in range(len(repeat_objects)) if index not in matched_repeat]
    return matches, unmatched_primary, unmatched_repeat


def verify_comparison_provenance(
    primary_state_path: Path, repeat_state_path: Path, repeat_selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fail closed unless both completed states describe the same isolated repeat sources."""
    primary_state_path = Path(primary_state_path)
    repeat_state_path = Path(repeat_state_path)
    repeat_selection_path = Path(repeat_selection_path)
    if primary_state_path.resolve() == repeat_state_path.resolve():
        raise ValueError("Primary and repeat artifacts must be distinct files.")
    primary = load_review_state(primary_state_path)
    repeat = load_review_state(repeat_state_path)
    selection = json.loads(repeat_selection_path.read_text(encoding="utf-8"))
    validate_review_state(primary)
    validate_review_state(repeat)
    repeat_ids = list(repeat.get("review_scope", {}).get("image_ids", []))
    if primary.get("review_scope", {}).get("name") != "calibration":
        raise ValueError("Primary comparison artifact must have calibration scope.")
    if repeat.get("review_scope", {}).get("name") != "calibration_repeat" or len(repeat_ids) != 8:
        raise ValueError("Repeat comparison artifact must have exactly eight calibration_repeat images.")
    if selection.get("image_ids") != repeat_ids or selection.get("target_size") != 8:
        raise ValueError("Repeat selection artifact does not match the eight-image repeat scope.")
    if not set(repeat_ids) <= set(primary["review_scope"]["image_ids"]):
        raise ValueError("Repeat scope contains images outside the primary calibration scope.")
    primary_sha256 = sha256_file(primary_state_path)
    if selection.get("source_primary_state_sha256") != primary_sha256:
        raise ValueError("Repeat selection provenance does not match the primary artifact hash.")
    if primary["protocol"] != repeat["protocol"] or repeat.get("repeat_review", {}).get("source_protocol_version") != primary["protocol"]["version"]:
        raise ValueError("Primary and repeat protocol provenance differs.")
    if primary["component_review"]["component_manifest_sha256"] != repeat["component_review"]["component_manifest_sha256"]:
        raise ValueError("Primary and repeat component manifest provenance differs.")
    source_mismatch = [
        image_id for image_id in repeat_ids
        if any(primary["images"][image_id][field] != repeat["images"][image_id][field] for field in SOURCE_FIELDS)
    ]
    if source_mismatch:
        raise ValueError(f"Primary and repeat source-image provenance differs: {source_mismatch}")
    unfinished = [
        image_id for image_id in repeat_ids
        if repeat["images"][image_id]["review_status"] != "reviewed"
        or unresolved_candidate_component_ids(repeat, image_id)
        or unresolved_candidate_component_ids(primary, image_id)
    ]
    if unfinished:
        raise ValueError(f"Comparison requires complete primary and repeat coverage: {unfinished}")
    provenance = {
        "primary_state_path": str(primary_state_path),
        "primary_state_sha256": primary_sha256,
        "repeat_state_path": str(repeat_state_path),
        "repeat_state_sha256": sha256_file(repeat_state_path),
        "artifacts_distinct": True,
        "repeat_scope_name": repeat["review_scope"]["name"],
        "repeat_scope_size": len(repeat_ids),
        "image_ids": repeat_ids,
        "protocol_version": primary["protocol"]["version"],
        "protocol_sha256": primary["protocol"]["sha256"],
        "component_manifest_sha256": primary["component_review"]["component_manifest_sha256"],
        "repeat_selection_version": selection["selection_version"],
        "repeat_selection_seed": selection["selection_seed"],
        "repeat_selection_method": "SHA-256 rank of calibration-repeat-v1:seed:image_id, then image ID tie-break",
        "selection_source_primary_sha256_matches": True,
        "source_image_provenance_matches": True,
        "repeat_annotations_are_isolated": primary["images"] is not repeat["images"],
    }
    return primary, repeat, selection, provenance


def _structure_summary(state: dict[str, Any], image_id: str) -> dict[str, Any]:
    image = state["images"][image_id]
    accepted_by_component: dict[int, list[str]] = defaultdict(list)
    for annotation in image["annotations"]:
        if annotation["annotation_status"] == "accepted":
            for component_id in annotation_component_ids(annotation):
                accepted_by_component[component_id].append(annotation["instance_id"])
    records = [
        {
            "resolution_type": record["resolution_type"],
            "source_candidate_component_ids": record["source_candidate_component_ids"],
            "resolved_annotation_instance_ids": record["resolved_annotation_instance_ids"],
            "reviewer_notes": record["reviewer_notes"],
        }
        for record in state["resolution_records"] if record["image_id"] == image_id
    ]
    return {
        "resolution_records": records,
        "accepted_instance_ids_by_component": {str(component_id): sorted(instance_ids) for component_id, instance_ids in sorted(accepted_by_component.items())},
    }


def analyze_intra_rater_consistency(
    primary_state_path: Path, repeat_state_path: Path, repeat_selection_path: Path,
) -> dict[str, Any]:
    """Produce a non-mutating intra-rater consistency report for the completed repeat."""
    primary_before = sha256_file(primary_state_path)
    repeat_before = sha256_file(repeat_state_path)
    primary, repeat, _, provenance = verify_comparison_provenance(
        primary_state_path, repeat_state_path, repeat_selection_path,
    )
    primary_state_digest = _canonical_digest(primary)
    repeat_state_digest = _canonical_digest(repeat)
    component_rows = []
    confusion = {primary_status: {repeat_status: 0 for repeat_status in DIRECT_DISPOSITIONS} for primary_status in DIRECT_DISPOSITIONS}
    transitions = Counter()
    structural_uncertainty_cases = []
    count_rows = []
    matched_objects = []
    unmatched_objects = []
    structure_discrepancies = []
    for image_id in provenance["image_ids"]:
        primary_image = primary["images"][image_id]
        repeat_image = repeat["images"][image_id]
        primary_components = component_dispositions(primary, image_id)
        repeat_components = component_dispositions(repeat, image_id)
        for component_id in primary_image["candidate_component_ids"]:
            primary_component = primary_components[component_id]
            repeat_component = repeat_components[component_id]
            row = {
                "image_id": image_id,
                "component_id": component_id,
                "primary_kind": primary_component["kind"],
                "repeat_kind": repeat_component["kind"],
                "primary_disposition": primary_component.get("disposition"),
                "repeat_disposition": repeat_component.get("disposition"),
                "primary_structure_reason": primary_component.get("reason"),
                "repeat_structure_reason": repeat_component.get("reason"),
            }
            row["comparable"] = primary_component["kind"] == repeat_component["kind"] == "direct"
            component_rows.append(row)
            if row["comparable"]:
                primary_status = row["primary_disposition"]
                repeat_status = row["repeat_disposition"]
                confusion[primary_status][repeat_status] += 1
                transitions[(primary_status, repeat_status)] += 1
            elif primary_component.get("disposition") == "uncertain" or repeat_component.get("disposition") == "uncertain":
                structural_uncertainty_cases.append(row)
        primary_structure = _structure_summary(primary, image_id)
        repeat_structure = _structure_summary(repeat, image_id)
        primary_count = sum(annotation["annotation_status"] == "accepted" for annotation in primary_image["annotations"])
        repeat_count = sum(annotation["annotation_status"] == "accepted" for annotation in repeat_image["annotations"])
        primary_uncertain = sum(annotation["annotation_status"] == "uncertain" for annotation in primary_image["annotations"])
        repeat_uncertain = sum(annotation["annotation_status"] == "uncertain" for annotation in repeat_image["annotations"])
        structure_changed = primary_structure != repeat_structure
        count_rows.append(
            {
                "image_id": image_id,
                "primary_accepted_instances": primary_count,
                "repeat_accepted_instances": repeat_count,
                "absolute_count_difference": abs(primary_count - repeat_count),
                "structure_changed": structure_changed,
                "primary_uncertain_annotations": primary_uncertain,
                "repeat_uncertain_annotations": repeat_uncertain,
                "uncertainty_change_cooccurs_with_count_difference": primary_uncertain != repeat_uncertain and primary_count != repeat_count,
            }
        )
        if structure_changed:
            structure_discrepancies.append(
                {
                    "image_id": image_id,
                    "primary_structure": primary_structure,
                    "repeat_structure": repeat_structure,
                    "primary_accepted_instances": primary_count,
                    "repeat_accepted_instances": repeat_count,
                }
            )
        matches, unmatched_primary, unmatched_repeat = match_accepted_objects(primary_image, repeat_image)
        for match in matches:
            matched_objects.append({"image_id": image_id, **match})
        for side, objects, opposite_objects in (
            ("primary", unmatched_primary, _accepted_objects(repeat_image)),
            ("repeat", unmatched_repeat, _accepted_objects(primary_image)),
        ):
            for item in objects:
                best_iou = max((mask_iou(item["mask"], other["mask"]) for other in opposite_objects), default=0.0)
                unmatched_objects.append(
                    {
                        "image_id": image_id,
                        "side": side,
                        "instance_id": item["annotation"]["instance_id"],
                        "source_candidate_component_ids": annotation_component_ids(item["annotation"]),
                        "decision": item["annotation"]["annotation_status"],
                        "reviewer_notes": item["annotation"]["reviewer_notes"],
                        "primary_accepted_instances": primary_count,
                        "repeat_accepted_instances": repeat_count,
                        "best_iou_against_opposite_pass": best_iou,
                    }
                )
    comparable_rows = [row for row in component_rows if row["comparable"]]
    exact_matches = sum(row["primary_disposition"] == row["repeat_disposition"] for row in comparable_rows)
    ious = [match["mask_iou"] for match in matched_objects]
    uncertainty_transitions = {
        "uncertain_in_both": transitions[("uncertain", "uncertain")],
        "uncertain_to_accepted": transitions[("uncertain", "accepted")],
        "uncertain_to_rejected_bedrock": transitions[("uncertain", "rejected_bedrock")],
        "uncertain_to_rejected_noise": transitions[("uncertain", "rejected_noise")],
        "accepted_to_uncertain": transitions[("accepted", "uncertain")],
        "rejected_bedrock_to_uncertain": transitions[("rejected_bedrock", "uncertain")],
        "rejected_noise_to_uncertain": transitions[("rejected_noise", "uncertain")],
        "structural_uncertainty_cases_excluded_from_categorical_agreement": structural_uncertainty_cases,
    }
    clarification_candidates = []
    if any(count["structure_changed"] for count in count_rows):
        clarification_candidates.append("Clarify when a candidate component should be split into multiple instances or merged with adjacent components, and require matching resolution records in both passes.")
    if any(row["primary_disposition"] == "accepted" and row["repeat_disposition"] == "rejected_bedrock" or row["primary_disposition"] == "rejected_bedrock" and row["repeat_disposition"] == "accepted" for row in comparable_rows):
        clarification_candidates.append("Clarify the visual boundary between a discrete rock and Bedrock for components that switch between accepted and rejected_bedrock.")
    if any(value for key, value in uncertainty_transitions.items() if key != "structural_uncertainty_cases_excluded_from_categorical_agreement") or structural_uncertainty_cases:
        clarification_candidates.append("Clarify the intended use of uncertain as a terminal exclusion, especially where it coexists with or transitions to a resolved component disposition.")
    if unmatched_objects:
        clarification_candidates.append("Clarify object-count and object-boundary handling for unmatched accepted instances; do not treat count agreement as polygon agreement.")
    recommendation = "CLARIFY" if clarification_candidates else "FREEZE"
    report = {
        "analysis_version": "intra-rater-consistency-v1",
        "analysis_type": "intra-rater consistency",
        "provenance": provenance,
        "component_disposition_agreement": {
            "total_candidate_components": len(component_rows),
            "comparable_components": len(comparable_rows),
            "structural_or_noncomparable_components": len(component_rows) - len(comparable_rows),
            "exact_disposition_matches": exact_matches,
            "exact_disposition_agreement_rate": exact_matches / len(comparable_rows) if comparable_rows else None,
            "confusion_counts": confusion,
            "component_rows": component_rows,
        },
        "accepted_instance_count_agreement": {
            "per_image": count_rows,
            "images_with_identical_counts": sum(row["absolute_count_difference"] == 0 for row in count_rows),
            "mean_absolute_count_difference": mean(row["absolute_count_difference"] for row in count_rows),
            "total_primary_instances": sum(row["primary_accepted_instances"] for row in count_rows),
            "total_repeat_instances": sum(row["repeat_accepted_instances"] for row in count_rows),
        },
        "accepted_object_geometry": {
            "matching_method": "Within-image deterministic maximum mask-IoU one-to-one Hungarian assignment; zero-IoU assignments are reported as unmatched.",
            "matched_object_count": len(matched_objects),
            "per_object": matched_objects,
            "mean_mask_iou": mean(ious) if ious else None,
            "median_mask_iou": median(ious) if ious else None,
            "min_mask_iou": min(ious, default=None),
            "max_mask_iou": max(ious, default=None),
        },
        "unmatched_objects": unmatched_objects,
        "uncertainty_consistency": uncertainty_transitions,
        "split_merge_discrepancies": structure_discrepancies,
        "protocol_clarification_candidates": clarification_candidates,
        "CALIBRATION_PROTOCOL_RECOMMENDATION": recommendation,
        "limitations": [
            "This is an eight-image intra-rater consistency sample, not inter-rater reliability or model evaluation.",
            "Raw disposition agreement is reported without chance-corrected statistics because the small, prevalence-skewed sample would make them unstable.",
            "Structured split/merge and multi-annotation components are excluded from simple categorical agreement rather than forced into a misleading category.",
            "No annotation is designated correct and no annotation artifact is modified by this analysis.",
        ],
    }
    if _canonical_digest(primary) != primary_state_digest or _canonical_digest(repeat) != repeat_state_digest:
        raise RuntimeError("Intra-rater analysis unexpectedly mutated an annotation state in memory.")
    if sha256_file(primary_state_path) != primary_before or sha256_file(repeat_state_path) != repeat_before:
        raise RuntimeError("Intra-rater analysis unexpectedly modified an annotation artifact.")
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for field, value in row.items()})


def write_analysis_outputs(report: dict[str, Any], output_dir: Path, markdown_path: Path) -> None:
    """Write stable machine-readable tables plus a concise research-facing report."""
    output_dir = Path(output_dir)
    markdown_path = Path(markdown_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "intra_rater_consistency.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "component_dispositions.csv", report["component_disposition_agreement"]["component_rows"])
    _write_csv(output_dir / "instance_counts.csv", report["accepted_instance_count_agreement"]["per_image"])
    _write_csv(output_dir / "matched_instances.csv", report["accepted_object_geometry"]["per_object"])
    _write_csv(output_dir / "unmatched_instances.csv", report["unmatched_objects"])
    component = report["component_disposition_agreement"]
    counts = report["accepted_instance_count_agreement"]
    geometry = report["accepted_object_geometry"]
    confusion_lines = ["| Primary \\ Repeat | Accepted | Rejected noise | Rejected Bedrock | Uncertain |", "| --- | ---: | ---: | ---: | ---: |"]
    for primary_status in DIRECT_DISPOSITIONS:
        row = component["confusion_counts"][primary_status]
        confusion_lines.append(
            f"| {primary_status} | {row['accepted']} | {row['rejected_noise']} | {row['rejected_bedrock']} | {row['uncertain']} |"
        )
    count_lines = ["| Image | Primary | Repeat | Absolute difference | Structure changed |", "| --- | ---: | ---: | ---: | --- |"]
    count_lines.extend(
        f"| {row['image_id']} | {row['primary_accepted_instances']} | {row['repeat_accepted_instances']} | {row['absolute_count_difference']} | {row['structure_changed']} |"
        for row in counts["per_image"]
    )
    geometry_lines = ["| Image | Primary instance | Repeat instance | Mask IoU | Primary area | Repeat area |", "| --- | --- | --- | ---: | ---: | ---: |"]
    geometry_lines.extend(
        f"| {row['image_id']} | {row['primary_instance_id']} | {row['repeat_instance_id']} | {row['mask_iou']:.4f} | {row['primary_area_pixels']} | {row['repeat_area_pixels']} |"
        for row in geometry["per_object"]
    )
    uncertainty = report["uncertainty_consistency"]
    uncertainty_lines = [
        f"- uncertain -> uncertain: {uncertainty['uncertain_in_both']}",
        f"- uncertain -> accepted: {uncertainty['uncertain_to_accepted']}",
        f"- uncertain -> rejected_bedrock: {uncertainty['uncertain_to_rejected_bedrock']}",
        f"- uncertain -> rejected_noise: {uncertainty['uncertain_to_rejected_noise']}",
        f"- accepted/rejected -> uncertain: {uncertainty['accepted_to_uncertain'] + uncertainty['rejected_bedrock_to_uncertain'] + uncertainty['rejected_noise_to_uncertain']}",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                "# Intra-Rater Consistency",
                "",
                "## Scope",
                "",
                "This protocol-stability analysis compares the completed primary calibration against its isolated eight-image repeat review. It is not inter-rater reliability, inter-annotator agreement, expert agreement, or model evaluation.",
                "",
                "## Provenance",
                "",
                f"- Primary artifact SHA-256: `{report['provenance']['primary_state_sha256']}`",
                f"- Repeat artifact SHA-256: `{report['provenance']['repeat_state_sha256']}`",
                f"- Protocol: `{report['provenance']['protocol_version']}` (`{report['provenance']['protocol_sha256']}`)",
                f"- Repeat selection: `{report['provenance']['repeat_selection_version']}`, seed `{report['provenance']['repeat_selection_seed']}`",
                f"- Images compared: {len(report['provenance']['image_ids'])}",
                "",
                "## Component Dispositions",
                "",
                f"- Directly comparable components: {component['comparable_components']} of {component['total_candidate_components']}",
                f"- Exact matches: {component['exact_disposition_matches']} ({component['exact_disposition_agreement_rate']:.1%})" if component["exact_disposition_agreement_rate"] is not None else "- Exact matches: no directly comparable components",
                f"- Structural/noncomparable components: {component['structural_or_noncomparable_components']}",
                "- The JSON artifact contains the raw primary-by-repeat disposition matrix; structured split/merge and multi-annotation components are not collapsed into categorical agreement.",
                "",
                "### Confusion Counts",
                "",
                *confusion_lines,
                "",
                "## Accepted Instances",
                "",
                f"- Identical accepted-instance counts: {counts['images_with_identical_counts']} of {len(counts['per_image'])} images",
                f"- Mean absolute count difference: {counts['mean_absolute_count_difference']:.3f}",
                f"- Total accepted instances: primary {counts['total_primary_instances']}; repeat {counts['total_repeat_instances']}",
                f"- Matched accepted objects: {geometry['matched_object_count']}",
                f"- Mask IoU: mean {geometry['mean_mask_iou']:.4f}, median {geometry['median_mask_iou']:.4f}, range {geometry['min_mask_iou']:.4f}-{geometry['max_mask_iou']:.4f}" if geometry["matched_object_count"] else "- Mask IoU: no matched accepted objects",
                "",
                "### Per-Image Counts",
                "",
                *count_lines,
                "",
                "### Matched Object Geometry",
                "",
                *geometry_lines,
                "",
                "## Discrepancies",
                "",
                f"- Unmatched accepted instances: {len(report['unmatched_objects'])}",
                f"- Split/merge or multi-annotation structure differences: {len(report['split_merge_discrepancies'])}",
                f"- Structured uncertainty cases excluded from simple categorical agreement: {len(report['uncertainty_consistency']['structural_uncertainty_cases_excluded_from_categorical_agreement'])}",
                f"- Structural discrepancy image IDs: {', '.join(item['image_id'] for item in report['split_merge_discrepancies']) or 'none'}",
                "- Per-instance unmatched evidence, including source components, best IoU, count context, and reviewer notes, is in `unmatched_instances.csv`.",
                "",
                "### Uncertainty Transitions",
                "",
                *uncertainty_lines,
                "",
                "## Recommendation",
                "",
                f"`CALIBRATION_PROTOCOL_RECOMMENDATION = {report['CALIBRATION_PROTOCOL_RECOMMENDATION']}`",
                "",
                *[f"- {item}" for item in report["protocol_clarification_candidates"]],
                "",
                "## Limitations",
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
    parser.add_argument("--repeat-selection-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--markdown-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_intra_rater_consistency(
        args.primary_state_path, args.repeat_state_path, args.repeat_selection_path,
    )
    write_analysis_outputs(report, args.output_dir, args.markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()