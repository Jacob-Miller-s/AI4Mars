import csv
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ai4mars.dataset import (
    _select_image_candidate_for_mask,
    find_image_files,
    find_mask_files,
    normalize_ai4mars_mask,
)


AI4MARS_DOI = "10.5281/zenodo.15995036"
AI4MARS_ARCHIVE_METADATA = {
    "ai4mars-dataset-merged-0.6": {
        "dataset_version": "ai4mars-dataset-merged-0.6",
        "archive_filename": "ai4mars-dataset-merged-0.6.zip",
        "archive_md5": "daf80a86021253292e6c425f97baa5c6",
        "archive_sha256": "",
        "dataset_doi": AI4MARS_DOI,
    },
    "ai4mars-labels-unmerged": {
        "dataset_version": "ai4mars-labels-unmerged",
        "archive_filename": "ai4mars-labels-unmerged.zip",
        "archive_md5": "49fc7a969dfddc0c06d0020edda432c2",
        "archive_sha256": "",
        "dataset_doi": AI4MARS_DOI,
    },
}

MANIFEST_FIELDNAMES = [
    "dataset_version",
    "dataset_doi",
    "archive_filename",
    "archive_md5",
    "archive_sha256",
    "dataset_relative_image_path",
    "dataset_relative_mask_path",
    "stable_source_image_id",
    "mission",
    "rover",
    "camera",
    "sequence_id",
    "label_role",
    "agreement_threshold",
    "label_scheme",
    "image_width",
    "image_height",
    "mask_width",
    "mask_height",
    "valid_pixel_fraction",
    "per_class_pixel_counts_json",
    "candidate_image_count",
    "stem_relationship",
    "exclusion_reason",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_dataset_root(raw_data_root: Path) -> Path:
    raw_data_root = Path(raw_data_root)
    candidates = sorted(raw_data_root.rglob("ai4mars-dataset-merged-0.6"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find extracted AI4Mars root under {raw_data_root}."
        )
    return candidates[0]


def infer_mission_rover_camera(path_obj: Path) -> Tuple[str, str, str]:
    parts = [part.lower() for part in path_obj.parts]
    mission = next((token for token in ("m2020", "msl", "mer") if token in parts), "unknown")
    rover_map = {
        "m2020": "perseverance",
        "msl": "curiosity",
        "mer": "spirit_opportunity",
    }
    rover = rover_map.get(mission, mission)
    camera = next(
        (token for token in ("ncam", "mcam", "zcam", "eff", "edr", "mxy", "hafiq") if token in parts),
        "unknown",
    )
    return mission, rover, camera


def infer_label_scheme(mask_relative_path: str) -> str:
    rel = mask_relative_path.lower()
    return "M2020_GEO" if "/m2020_geo/" in rel else "NAV"


def infer_label_role(mask_relative_path: str) -> str:
    rel = mask_relative_path.lower()
    if "/labels/test/" in rel:
        return "expert_gold_test"
    return "crowdsourced_train"


def infer_agreement_threshold(mask_relative_path: str) -> str:
    match = re.search(r"masked-gold-(min\d+-100agree)", mask_relative_path.lower())
    return match.group(1) if match else ""


def infer_sequence_id(stable_source_image_id: str) -> str:
    tokens = stable_source_image_id.split("_")
    for token in tokens:
        upper = token.upper()
        if upper.startswith("F") and any(cam in upper for cam in ("NCAM", "AUT", "EFF")):
            return upper
        if upper.startswith("N") and any(cam in upper for cam in ("NCAM", "MCAM", "ZCAM")):
            return upper
        if upper.startswith("2F") and "EFF" in upper:
            return upper
    if len(tokens) >= 4:
        return "_".join(tokens[:4])
    return stable_source_image_id


def _find_match_stem(mask_stem: str, images_by_stem: Dict[str, List[Path]]) -> Tuple[Optional[str], List[Path]]:
    segments = mask_stem.split("_")
    for cutoff in range(len(segments), 0, -1):
        candidate_stem = "_".join(segments[:cutoff])
        candidates = images_by_stem.get(candidate_stem)
        if candidates:
            return candidate_stem, candidates
    return None, []


def _stem_relationship(mask_stem: str, matched_stem: Optional[str]) -> str:
    if matched_stem is None:
        return "no_match"
    if mask_stem == matched_stem:
        return "exact"
    if mask_stem.startswith(matched_stem + "_"):
        suffix = mask_stem[len(matched_stem) + 1 :]
        return f"trimmed_suffix:{suffix}"
    return "prefix_match"


def _relative_or_empty(path_obj: Optional[Path], dataset_root: Path) -> str:
    if path_obj is None:
        return ""
    return Path(path_obj).relative_to(dataset_root).as_posix()


def _mask_statistics(mask_path: Path) -> Tuple[int, int, float, str]:
    with Image.open(mask_path) as mask_obj:
        mask_w, mask_h = mask_obj.size
        mask = np.array(mask_obj, dtype=np.int64)
    mask = normalize_ai4mars_mask(mask, mask_path)
    total_pixels = int(mask.size)
    valid_pixels = int((mask != 255).sum())
    unique, counts = np.unique(mask, return_counts=True)
    counts_dict = {str(int(class_id)): int(count) for class_id, count in zip(unique, counts)}
    valid_fraction = valid_pixels / total_pixels if total_pixels else 0.0
    return mask_w, mask_h, valid_fraction, json.dumps(counts_dict, sort_keys=True)


def _mask_manifest_row(
    mask_path: Path,
    dataset_root: Path,
    images_by_stem: Dict[str, List[Path]],
    archive_meta: Dict[str, str],
) -> Tuple[Dict[str, Any], Optional[Path], Optional[str]]:
    matched_stem, candidates = _find_match_stem(mask_path.stem, images_by_stem)
    relation = _stem_relationship(mask_path.stem, matched_stem)
    selected_image: Optional[Path] = None
    exclusion_reason = ""
    image_w = ""
    image_h = ""

    if matched_stem is None:
        exclusion_reason = "unmatched_mask_no_candidate_image"
    else:
        try:
            selected_image = _select_image_candidate_for_mask(mask_path, candidates)
        except RuntimeError:
            exclusion_reason = "ambiguous_image_match"

    mask_relative_path = mask_path.relative_to(dataset_root).as_posix()
    mask_w, mask_h, valid_fraction, counts_json = _mask_statistics(mask_path)

    mission, rover, camera = infer_mission_rover_camera(mask_path)
    stable_source_image_id = matched_stem or ""
    sequence_id = infer_sequence_id(stable_source_image_id) if stable_source_image_id else ""

    if selected_image is not None:
        with Image.open(selected_image) as image_obj:
            image_w, image_h = image_obj.size
        if (image_w, image_h) != (mask_w, mask_h):
            exclusion_reason = "shape_mismatch"

    row = {
        "dataset_version": archive_meta["dataset_version"],
        "dataset_doi": archive_meta["dataset_doi"],
        "archive_filename": archive_meta["archive_filename"],
        "archive_md5": archive_meta["archive_md5"],
        "archive_sha256": archive_meta["archive_sha256"],
        "dataset_relative_image_path": _relative_or_empty(selected_image, dataset_root),
        "dataset_relative_mask_path": mask_relative_path,
        "stable_source_image_id": stable_source_image_id,
        "mission": mission,
        "rover": rover,
        "camera": camera,
        "sequence_id": sequence_id,
        "label_role": infer_label_role(mask_relative_path),
        "agreement_threshold": infer_agreement_threshold(mask_relative_path),
        "label_scheme": infer_label_scheme(mask_relative_path),
        "image_width": image_w,
        "image_height": image_h,
        "mask_width": mask_w,
        "mask_height": mask_h,
        "valid_pixel_fraction": f"{valid_fraction:.8f}",
        "per_class_pixel_counts_json": counts_json,
        "candidate_image_count": len(candidates),
        "stem_relationship": relation,
        "exclusion_reason": exclusion_reason,
    }
    ambiguous_mask = mask_relative_path if exclusion_reason == "ambiguous_image_match" else None
    return row, selected_image, ambiguous_mask


def build_dataset_manifest_rows(dataset_root: Path) -> List[Dict[str, Any]]:
    dataset_root = Path(dataset_root)
    archive_meta = AI4MARS_ARCHIVE_METADATA.get(
        dataset_root.name,
        {
            "dataset_version": dataset_root.name,
            "archive_filename": "",
            "archive_md5": "",
            "archive_sha256": "",
            "dataset_doi": AI4MARS_DOI,
        },
    )

    image_files = find_image_files(dataset_root)
    mask_files = find_mask_files(dataset_root)
    images_by_stem: Dict[str, List[Path]] = {}
    for image_path in image_files:
        images_by_stem.setdefault(image_path.stem, []).append(image_path)

    rows: List[Dict[str, Any]] = []
    selected_images: set[Path] = set()
    ambiguous_masks: List[str] = []
    mask_paths = sorted(mask_files)
    max_workers = min(32, max(4, (os.cpu_count() or 4)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _mask_manifest_row,
            mask_paths,
            [dataset_root] * len(mask_paths),
            [images_by_stem] * len(mask_paths),
            [archive_meta] * len(mask_paths),
        )
        for row, selected_image, ambiguous_mask in results:
            rows.append(row)
            if selected_image is not None:
                selected_images.add(selected_image)
            if ambiguous_mask is not None:
                ambiguous_masks.append(ambiguous_mask)

    for image_path in sorted(image_files):
        if image_path in selected_images:
            continue
        mission, rover, camera = infer_mission_rover_camera(image_path)
        stable_source_image_id = image_path.stem
        sequence_id = infer_sequence_id(stable_source_image_id)
        with Image.open(image_path) as image_obj:
            image_w, image_h = image_obj.size
        rows.append(
            {
                "dataset_version": archive_meta["dataset_version"],
                "dataset_doi": archive_meta["dataset_doi"],
                "archive_filename": archive_meta["archive_filename"],
                "archive_md5": archive_meta["archive_md5"],
                "archive_sha256": archive_meta["archive_sha256"],
                "dataset_relative_image_path": image_path.relative_to(dataset_root).as_posix(),
                "dataset_relative_mask_path": "",
                "stable_source_image_id": stable_source_image_id,
                "mission": mission,
                "rover": rover,
                "camera": camera,
                "sequence_id": sequence_id,
                "label_role": "",
                "agreement_threshold": "",
                "label_scheme": "",
                "image_width": image_w,
                "image_height": image_h,
                "mask_width": "",
                "mask_height": "",
                "valid_pixel_fraction": "",
                "per_class_pixel_counts_json": "",
                "candidate_image_count": 0,
                "stem_relationship": "unused_image",
                "exclusion_reason": "unmatched_image_unused",
            }
        )

    if ambiguous_masks:
        preview = ", ".join(ambiguous_masks[:3])
        raise RuntimeError(
            "Ambiguous image/mask matches detected. Fail-closed pairing requires manual resolution. "
            f"Example masks: {preview}"
        )

    return sorted(
        rows,
        key=lambda row: (
            row["mission"],
            row["camera"],
            row["stable_source_image_id"],
            row["dataset_relative_mask_path"],
            row["dataset_relative_image_path"],
        ),
    )


def write_dataset_manifest(dataset_root: Path, manifest_path: Path) -> List[Dict[str, Any]]:
    rows = build_dataset_manifest_rows(dataset_root)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    return rows


def _split_group_key(row: Dict[str, str]) -> str:
    return row.get("sequence_id") or row.get("stable_source_image_id")


def _assign_groups_to_train_val(groups: Dict[str, List[Dict[str, str]]], train_ratio: float, seed: int) -> Dict[str, List[Dict[str, str]]]:
    ordered_keys = sorted(
        groups,
        key=lambda key: hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    train_cutoff = int(round(len(ordered_keys) * train_ratio))
    if len(ordered_keys) >= 2:
        train_cutoff = min(max(train_cutoff, 1), len(ordered_keys) - 1)
    partitions = {"train": [], "val": []}
    for index, group_key in enumerate(ordered_keys):
        split_name = "train" if index < train_cutoff else "val"
        partitions[split_name].extend(groups[group_key])
    return partitions


def hash_identifier_rows(rows: Sequence[Dict[str, str]]) -> str:
    payload = "\n".join(
        "|".join(
            [
                row.get("dataset_relative_image_path", ""),
                row.get("dataset_relative_mask_path", ""),
                row.get("stable_source_image_id", ""),
                row.get("sequence_id", ""),
                row.get("split_name", ""),
            ]
        )
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_zero_overlap(split_rows: Dict[str, List[Dict[str, str]]]) -> None:
    source_ids = {name: {row["stable_source_image_id"] for row in rows} for name, rows in split_rows.items()}
    sequence_ids = {name: {row["sequence_id"] for row in rows} for name, rows in split_rows.items()}
    if "train" in split_rows and "val" in split_rows:
        if source_ids["train"] & source_ids["val"]:
            raise RuntimeError("Source-image overlap detected between train and val.")
        if sequence_ids["train"] & sequence_ids["val"]:
            raise RuntimeError("Sequence overlap detected between train and val.")

    protected_splits = [name for name in split_rows if name.startswith("test_")]
    for right in protected_splits:
        for left in ("train", "val"):
            if left not in split_rows:
                continue
            if source_ids[left] & source_ids[right]:
                raise RuntimeError(f"Source-image overlap detected between {left} and {right}.")
            if sequence_ids[left] & sequence_ids[right]:
                raise RuntimeError(f"Sequence overlap detected between {left} and {right}.")


def _row_matches_scope(
        row: Dict[str, str],
        mission_filter: Optional[str] = None,
        camera_filter: Optional[str] = None,
        ) -> bool:
        if (
            mission_filter is not None
            and row.get("mission") != mission_filter
        ):
            return False
        if (
            camera_filter is not None
            and row.get("camera") != camera_filter
        ):
            return False

        return True


def build_split_manifests(
    dataset_manifest_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    label_scheme: str = "NAV",
    mission_filter: Optional[str] = None,
    camera_filter: Optional[str] = None,
) -> Dict[str, Path]:
    rows = read_manifest_rows(dataset_manifest_path)


    included_rows = [
        row
        for row in rows
        if not (row.get("exclusion_reason") or "").strip()
        and _row_matches_scope(row,
                               mission_filter=mission_filter,
                               camera_filter=camera_filter,
            )
        ]

    expert_rows = [
        row
        for row in included_rows
        if row.get("label_role") == "expert_gold_test"
        and row.get("label_scheme") == label_scheme
    ]

    reserved_source_ids = {
        row.get("stable_source_image_id", "")
        for row in expert_rows
        if row.get("stable_source_image_id")
    }
    reserved_sequence_ids = {
        row.get("sequence_id", "")
        for row in expert_rows
        if row.get("sequence_id")
    }

    training_pool = [
        row
        for row in included_rows
        if row.get("label_role") == "crowdsourced_train"
        and row.get("label_scheme") == label_scheme
        and row.get("stable_source_image_id", "") not in reserved_source_ids
        and row.get("sequence_id", "") not in reserved_sequence_ids
    ]
    if not training_pool:
        raise RuntimeError("No crowdsourced training rows available for split generation.")

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in training_pool:
        grouped.setdefault(_split_group_key(row), []).append(dict(row))

    train_val = _assign_groups_to_train_val(grouped, train_ratio=train_ratio, seed=seed)
    split_rows: Dict[str, List[Dict[str, str]]] = {}
    for split_name, rows_for_split in train_val.items():
        split_rows[split_name] = []
        for row in sorted(rows_for_split, key=lambda item: (item["sequence_id"], item["stable_source_image_id"], item["dataset_relative_mask_path"])):
            row_copy = {key: value for key, value in row.items() if key != "exclusion_reason"}
            row_copy["split_name"] = split_name
            split_rows[split_name].append(row_copy)

    thresholds = sorted({row.get("agreement_threshold", "") for row in expert_rows if row.get("agreement_threshold")})
    for threshold in thresholds:
        split_name = f"test_{threshold.replace('-', '_')}"
        split_rows[split_name] = []
        for row in sorted(
            (item for item in expert_rows if item.get("agreement_threshold") == threshold),
            key=lambda item: (item["sequence_id"], item["stable_source_image_id"], item["dataset_relative_mask_path"]),
        ):
            row_copy = {key: value for key, value in row.items() if key != "exclusion_reason"}
            row_copy["split_name"] = split_name
            split_rows[split_name].append(row_copy)

    overlap_check_rows = {
        name: rows_for_split
        for name, rows_for_split in split_rows.items()
        if name in {"train", "val"} or name.startswith("test_")
    }
    _assert_zero_overlap(overlap_check_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, Path] = {}
    split_hashes: Dict[str, str] = {}
    split_fieldnames = [field for field in MANIFEST_FIELDNAMES if field != "exclusion_reason"] + ["split_name"]
    for split_name, rows_for_split in split_rows.items():
        split_path = output_dir / f"{split_name.lower()}_{label_scheme.lower()}.csv"
        with split_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=split_fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_split)
        split_paths[split_name] = split_path
        split_hashes[split_name] = hash_identifier_rows(rows_for_split)

    with (output_dir / f"split_manifest_hashes_{label_scheme.lower()}.json").open("w", encoding="utf-8") as fp:
        json.dump(split_hashes, fp, indent=2, sort_keys=True)

    return split_paths


def requirements_hash(requirements_path: Path) -> str:
    requirements_path = Path(requirements_path)
    if not requirements_path.exists():
        return ""
    return sha256_file(requirements_path)


def current_git_commit(project_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except Exception:
        return ""


def build_checkpoint_metadata(
    *,
    project_root: Path,
    dataset_manifest_path: Path,
    split_manifest_paths: Dict[str, Path],
    active_split_name: str,
    preprocessing: Dict[str, Any],
    loss_name: str,
    loss_weights: Sequence[float],
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    split_hashes = {name: sha256_file(path) for name, path in split_manifest_paths.items()}
    return {
        "dataset_manifest_path": Path(dataset_manifest_path).name,
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "split_manifest_hashes": split_hashes,
        "active_split_name": active_split_name,
        "git_commit_sha": current_git_commit(project_root),
        "preprocessing": preprocessing,
        "loss_name": loss_name,
        "loss_weights": list(loss_weights),
        "model_name": model_name,
        "seed": seed,
        "dependency_lock_hash": requirements_hash(Path(project_root) / "requirements.txt"),
        "python_version": sys.version,
        "platform": platform_string(),
    }


def platform_string() -> str:
    return f"{sys.platform} | {os.name}"


def write_run_record(
    run_dir: Path,
    *,
    config: Dict[str, Any],
    dataset_manifest_path: Path,
    split_manifest_paths: Dict[str, Path],
    metrics: Dict[str, Any],
    environment_text: str,
) -> None:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "dataset_manifest_hash.txt").write_text(sha256_file(dataset_manifest_path), encoding="utf-8")
    split_hashes = {name: sha256_file(path) for name, path in split_manifest_paths.items()}
    (run_dir / "split_manifest_hashes.json").write_text(
        json.dumps(split_hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "environment.txt").write_text(environment_text, encoding="utf-8")