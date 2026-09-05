"""
ai4mars/dataset.py
==============
PyTorch Dataset class for the AI4Mars terrain segmentation task.

AI4Mars label IDs (verify against official documentation!):
    0  -> soil
    1  -> bedrock
    2  -> sand
    3  -> big_rock
    255 -> ignore / unlabeled

The dataset expects image/mask pairs where:
  - Images are color rover photographs (JPG or PNG).
  - Masks are single-channel PNG files where each pixel value is a class ID.

Usage
-----
    from ai4mars.dataset import AI4MarsDataset, build_pairs_by_stem
    from ai4mars.data_paths import RAW_DATA_DIR

    image_files = find_image_files(RAW_DATA_DIR)
    mask_files  = find_mask_files(RAW_DATA_DIR)
    pairs       = build_pairs_by_stem(image_files, mask_files)
    dataset     = AI4MarsDataset(pairs)
    image, mask = dataset[0]  # image: [3, H, W] float32, mask: [H, W] int64
"""

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import csv
import hashlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _is_legacy_nav_ignore_mask(mask_path: Optional[Path]) -> bool:
    """Return True for NAV masks that use legacy ignore label 4 instead of 255.

    In the merged AI4Mars archive, MSL MastCam training masks under
    ``msl/mcam/labels/train`` use filenames like ``*_15033_merged.png`` and can
    contain the value ``4`` where the rest of the NAV dataset uses ``255`` for
    ignored / unlabeled pixels.
    """
    if mask_path is None:
        return False

    normalized_path = str(mask_path).replace("\\", "/").lower()
    return (
        "msl/mcam/labels/train" in normalized_path
        and normalized_path.endswith("_15033_merged.png")
    )


def normalize_ai4mars_mask(mask: np.ndarray, mask_path: Optional[Path] = None) -> np.ndarray:
    """Normalize dataset-specific mask-ID quirks to the canonical AI4Mars IDs.

    Parameters
    ----------
    mask : np.ndarray
        Integer class-ID mask.
    mask_path : Path, optional
        Source path for the mask. Used to detect known subset-specific quirks.

    Returns
    -------
    np.ndarray
        Normalized mask. The returned array is safe to use for training and
        visual inspection with the canonical label IDs.
    """
    if _is_legacy_nav_ignore_mask(mask_path) and np.any(mask == 4):
        mask = mask.copy()
        mask[mask == 4] = 255
    return mask


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class AI4MarsDataset(Dataset):
    """PyTorch Dataset for AI4Mars image/mask pairs.

    Parameters
    ----------
    pairs : list of (image_path, mask_path)
        Each element is a 2-tuple of path-like objects pointing to an image
        and its corresponding mask.
    image_size : tuple of (width, height)
        Target spatial resolution for both image and mask.
        PIL uses (width, height) ordering — note this is the *opposite* of
        numpy's (rows, cols) / (height, width) convention.
    transform : callable, optional
        An optional additional transform applied to the ``(image, mask)``
        pair *after* the standard tensor conversion.  Leave as ``None`` for
        the baseline.
    """

    def __init__(
        self,
        pairs: List[Tuple],
        image_size: Tuple[int, int] = (256, 256),
        transform=None,
        require_original_shape_match: bool = False,
        normalization_mean: Optional[Tuple[float, float, float]] = None,
        normalization_std: Optional[Tuple[float, float, float]] = None,
    ):
        # Ensure all paths are proper Path objects so that we can call
        # .exists(), .stem, etc. safely later.
        self.pairs = [(Path(img), Path(mask)) for img, mask in pairs]
        self.image_size = image_size  # (width, height)
        self.transform = transform
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("normalization_mean and normalization_std must be provided together.")
        if normalization_mean is not None:
            if len(normalization_mean) != 3 or len(normalization_std) != 3:
                raise ValueError("Image normalization requires three channel means and standard deviations.")
            if any(value <= 0 for value in normalization_std):
                raise ValueError("Image normalization standard deviations must be positive.")
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        if require_original_shape_match:
            mismatches = find_shape_mismatches(self.pairs)
            if mismatches:
                example = mismatches[0]
                raise ValueError(
                    "Found image/mask geometry mismatch while strict shape matching is enabled. "
                    f"Example pair: {example[0]} ({example[2][0]}x{example[2][1]}) vs "
                    f"{example[1]} ({example[3][0]}x{example[3][1]})."
                )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]

        # ------------------------------------------------------------------
        # Load raw image and mask from disk
        # ------------------------------------------------------------------
        # convert("RGB") ensures we always get 3 channels regardless of
        # whether the original file is grayscale, RGBA, palette-mode, etc.
        image = Image.open(image_path).convert("RGB")

        # Masks store integer class IDs — do NOT convert to RGB or the pixel
        # values will be changed.  Open as-is ("L" or "P" mode is fine).
        mask = Image.open(mask_path)

        # ------------------------------------------------------------------
        # Resize
        # ------------------------------------------------------------------
        # Standard BILINEAR resampling (PIL default) is fine for color images.
        image = image.resize(self.image_size, resample=Image.BILINEAR)

        # NEAREST-NEIGHBOUR is *required* for masks because the pixel values
        # are discrete class IDs (0, 1, 2, 3, 255).  Interpolating them would
        # create invalid intermediate values such as 1.7 or 127.
        mask = mask.resize(self.image_size, resample=Image.NEAREST)

        # ------------------------------------------------------------------
        # Convert to numpy arrays
        # ------------------------------------------------------------------
        # Image: H x W x 3, float32, range [0.0, 1.0]
        image = np.array(image, dtype=np.float32) / 255.0

        # Mask: H x W, int64, values are class IDs.
        # Some MSL MastCam NAV train masks use a legacy ignore label 4 instead
        # of the canonical 255; normalize them here so training/evaluation code
        # consistently sees the same ignore_index.
        mask = np.array(mask, dtype=np.int64)
        mask = normalize_ai4mars_mask(mask, mask_path)

        # ------------------------------------------------------------------
        # Convert to PyTorch tensors
        # ------------------------------------------------------------------
        # permute(2, 0, 1) converts H x W x C  →  C x H x W
        # which is the standard PyTorch channel-first format expected by
        # Conv2d layers.
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask = torch.from_numpy(mask).long()               # [H, W]

        if self.normalization_mean is not None and self.normalization_std is not None:
            mean = torch.tensor(self.normalization_mean, dtype=image.dtype).view(3, 1, 1)
            std = torch.tensor(self.normalization_std, dtype=image.dtype).view(3, 1, 1)
            image = (image - mean) / std

        # ------------------------------------------------------------------
        # Optional extra transforms
        # ------------------------------------------------------------------
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def find_image_files(root: Path) -> List[Path]:
    """Recursively find all image files under *root*.

    Searches for common image extensions used by rover cameras.

    .. note::
        ``.png`` is intentionally **excluded** here.  In the AI4Mars dataset,
        every actual rover photograph is ``.jpg``/``.jpeg``/``.tif`` — all
        ``.png`` files are masks (segmentation labels, rover masks, or range
        masks).  Including ``.png`` here would make every mask also match as
        an "image", corrupting the stem lookup used by
        :func:`build_pairs_by_stem`.

    Parameters
    ----------
    root : Path
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted list of image file paths.
    """
    image_extensions = {".jpg", ".jpeg", ".tif", ".tiff"}
    files = [
        p for p in root.rglob("*")
        if p.suffix.lower() in image_extensions and p.is_file()
    ]
    return sorted(files)


def find_mask_files(root: Path) -> List[Path]:
    """Recursively find all mask files under *root*.

    AI4Mars masks are PNG files stored under a ``labels/`` directory (e.g.
    ``labels/train``, ``labels/test/masked-gold-min1-100agree``,
    ``labels/NAV``, ``labels/M2020_GEO``).  We deliberately restrict the
    search to paths containing a ``labels`` directory component because the
    merged AI4Mars archive also ships auxiliary ``.png`` files under
    ``images/`` (rover masks in ``mxy/`` and 30m range masks in
    ``rng-30m/``) that are **not** segmentation labels and would otherwise be
    misidentified as masks.

    Parameters
    ----------
    root : Path
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted list of mask file paths.
    """
    files = [
        p for p in root.rglob("*.png")
        if p.is_file() and "labels" in p.parts
    ]
    return sorted(files)


def build_pairs_by_stem(
    image_files: List[Path],
    mask_files: List[Path],
) -> List[Tuple[Path, Path]]:
    """Pair image and mask files by matching their filename stems.

    A "stem" is the filename without its extension.  Per the AI4Mars
    documentation (``info.md``): *"Names of images match names of labels,
    except for the extension (JPG, PNG) and sometimes an obvious suffix
    (e.g. ``_merged``)"*.  In practice the suffix varies by camera/rover:

    - MSL NavCam (``ncam``):  ``<stem>.JPG``      ↔ ``<stem>.png``            (exact match)
    - MSL MastCam (``mcam``): ``<stem>.JPG``      ↔ ``<stem>_15033_merged.png``
    - MER (``eff``):          ``<stem>.JPG``      ↔ ``<stem>_merged6.png``
    - M2020 tiled labels:     ``<stem>.jpeg``     ↔ ``<stem>_01_195J_merged3.png``

    To handle all of these with one rule, a mask is matched to an image by
    progressively trimming trailing ``_``-delimited segments off the mask's
    stem until the remainder equals a known image stem (the *longest*
    matching prefix wins).  This also naturally supports the tiled M2020
    labels, where several mask files (one per tile/quadrant) can map back to
    the same source image.

    Parameters
    ----------
    image_files : list[Path]
        Candidate image paths (from :func:`find_image_files`).
    mask_files : list[Path]
        Candidate mask paths (from :func:`find_mask_files`).

    Returns
    -------
    list[tuple[Path, Path]]
        Matched (image, mask) pairs.  Masks with no matching image stem are
        silently skipped.
    """
    # Build a stem → list[path] lookup. Some merged AI4Mars subsets contain
    # duplicate stems under different folders (for example mer/images/eff and
    # mer/images/test). We keep all candidates and disambiguate per mask.
    images_by_stem: Dict[str, List[Path]] = {}
    for image_path in image_files:
        images_by_stem.setdefault(image_path.stem, []).append(image_path)

    pairs: List[Tuple[Path, Path]] = []
    for mask_path in mask_files:
        img_path = _find_source_image_for_mask(mask_path, images_by_stem)
        if img_path is not None:
            pairs.append((img_path, mask_path))

    return pairs


def find_duplicate_image_stems(image_files: List[Path]) -> Dict[str, List[Path]]:
    """Return stems that appear in more than one image path."""
    images_by_stem: Dict[str, List[Path]] = {}
    for image_path in image_files:
        images_by_stem.setdefault(image_path.stem, []).append(image_path)
    return {stem: paths for stem, paths in images_by_stem.items() if len(paths) > 1}


def group_pairs_by_source_stem(pairs: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
    """Group image/mask pairs by the stem of the source image.

    This guarantees that all labels derived from the same source image stay in
    the same partition when building train/validation/test splits.
    """
    grouped: Dict[str, List[Tuple[Path, Path]]] = {}
    for image_path, mask_path in pairs:
        source_stem = Path(image_path).stem
        grouped.setdefault(source_stem, []).append((Path(image_path), Path(mask_path)))
    return grouped


def split_pairs_by_source_stem(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Deterministically split pairs into train/val/test partitions.

    The split is metadata-aware and immutable for a fixed input dataset and
    seed: every pair from the same source-image stem is assigned to the same
    partition.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    grouped = group_pairs_by_source_stem(pairs)
    ordered_stems = sorted(
        grouped.keys(),
        key=lambda stem: hashlib.sha1(f"{seed}:{stem}".encode("utf-8")).hexdigest(),
    )

    total_groups = len(ordered_stems)
    if total_groups == 0:
        return {"train": [], "val": [], "test": []}

    target_counts = [
        int(round(total_groups * train_ratio)),
        int(round(total_groups * val_ratio)),
        int(round(total_groups * test_ratio)),
    ]

    # Fix rounding drift so the counts add up exactly.
    target_counts[0] += total_groups - sum(target_counts)

    # Ensure every partition is populated when enough groups exist.
    if total_groups >= 3:
        for idx in range(3):
            if target_counts[idx] < 1:
                donor = max(range(3), key=lambda j: target_counts[j])
                if target_counts[donor] > 1:
                    target_counts[donor] -= 1
                    target_counts[idx] = 1

    while sum(target_counts) < total_groups:
        donor = max(range(3), key=lambda j: target_counts[j])
        target_counts[donor] += 1

    while sum(target_counts) > total_groups:
        donor = max(range(3), key=lambda j: target_counts[j])
        if target_counts[donor] > 1:
            target_counts[donor] -= 1
        else:
            break

    train_cutoff = target_counts[0]
    val_cutoff = train_cutoff + target_counts[1]

    partitions: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    for index, stem in enumerate(ordered_stems):
        if index < train_cutoff:
            split = "train"
        elif index < val_cutoff:
            split = "val"
        else:
            split = "test"
        partitions[split].extend(grouped[stem])

    return partitions


def find_shape_mismatches(
    pairs: List[Tuple[Path, Path]],
) -> List[Tuple[Path, Path, Tuple[int, int], Tuple[int, int]]]:
    """Return pairs whose original image and mask dimensions differ.

    Returns
    -------
    list[tuple[Path, Path, tuple[int, int], tuple[int, int]]]
        Each tuple contains (image_path, mask_path, image_size_wh, mask_size_wh).
    """
    mismatches: List[Tuple[Path, Path, Tuple[int, int], Tuple[int, int]]] = []
    for image_path, mask_path in pairs:
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        with Image.open(image_path) as image_obj:
            image_size = image_obj.size
        with Image.open(mask_path) as mask_obj:
            mask_size = mask_obj.size
        if image_size != mask_size:
            mismatches.append((image_path, mask_path, image_size, mask_size))
    return mismatches


def resolve_dataset_relative_path(dataset_root: Path, stored_path: str) -> Path:
    """Resolve a canonical manifest path while preventing dataset-root escape."""
    value = (stored_path or "").strip()
    if not value:
        raise ValueError("Manifest path cannot be empty.")

    posix_path = PurePosixPath(value.replace("\\", "/"))
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"Manifest path must be dataset-relative, not absolute: {value!r}")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError(f"Manifest path cannot traverse outside the dataset root: {value!r}")

    root = Path(dataset_root).resolve()
    resolved = (root / Path(*posix_path.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes dataset root: {value!r}") from error
    return resolved


def load_pairs_from_manifest(
    manifest_path: Path,
    require_existing_files: bool = True,
    require_shape_match: bool = True,
    required_label_scheme: Optional[str] = None,
    dataset_root: Optional[Path] = None,
) -> List[Tuple[Path, Path]]:
    """Load image/mask pairs from a CSV manifest.

        The manifest must contain a mask-path column and an image-path column.
    Accepted aliases:
            - image column: ``selected_image_path`` or ``image_path`` or
                ``dataset_relative_image_path``
            - mask column: ``mask_path`` or ``dataset_relative_mask_path``

    Optional filters:
      - ``required_label_scheme`` checks the ``label_scheme`` column when present.
      - ``require_shape_match`` checks ``shape_match`` when present and always
        validates original geometry from disk.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")

    image_column = None
    for candidate in ("selected_image_path", "image_path", "dataset_relative_image_path"):
        if candidate in rows[0]:
            image_column = candidate
            break
    if image_column is None:
        raise ValueError(
            "Manifest must contain one of the image columns: "
            "'selected_image_path', 'image_path', or 'dataset_relative_image_path'."
        )
    mask_column = None
    for candidate in ("mask_path", "dataset_relative_mask_path"):
        if candidate in rows[0]:
            mask_column = candidate
            break
    if mask_column is None:
        raise ValueError("Manifest must contain 'mask_path' or 'dataset_relative_mask_path' column.")

    pairs: List[Tuple[Path, Path]] = []
    for row in rows:
        if required_label_scheme is not None and "label_scheme" in row:
            if (row.get("label_scheme") or "").strip() != required_label_scheme:
                continue

        if require_shape_match and "shape_match" in row:
            shape_match_text = (row.get("shape_match") or "").strip()
            if shape_match_text in {"0", "False", "false"}:
                continue

        exclusion_reason = (row.get("exclusion_reason") or "").strip()
        if exclusion_reason:
            continue

        image_text = (row.get(image_column) or "").strip()
        mask_text = (row.get(mask_column) or "").strip()
        image_path = Path(image_text)
        mask_path = Path(mask_text)
        if dataset_root is not None:
            if image_text:
                image_path = resolve_dataset_relative_path(dataset_root, image_text)
            if mask_text:
                mask_path = resolve_dataset_relative_path(dataset_root, mask_text)
        if not image_path or not mask_path:
            continue
        if require_existing_files and (not image_path.exists() or not mask_path.exists()):
            raise FileNotFoundError(
                "Manifest references missing files: "
                f"image={image_path} mask={mask_path}"
            )
        pairs.append((image_path, mask_path))

    if not pairs:
        raise ValueError(
            "No usable pairs found in manifest after filtering. "
            f"Manifest: {manifest_path}"
        )

    if require_shape_match:
        mismatches = find_shape_mismatches(pairs)
        if mismatches:
            example = mismatches[0]
            raise ValueError(
                "Manifest contains image/mask pairs with mismatched original geometry. "
                f"Example pair: {example[0]} ({example[2][0]}x{example[2][1]}) vs "
                f"{example[1]} ({example[3][0]}x{example[3][1]})."
            )

    return pairs


def _select_image_candidate_for_mask(mask_path: Path, candidates: List[Path]) -> Path:
    """Choose the most plausible source image when multiple stems match."""
    if len(candidates) == 1:
        return candidates[0]

    mask_parts = [part.lower() for part in mask_path.parts]

    def score_candidate(image_path: Path) -> int:
        score = 0
        image_parts = [part.lower() for part in image_path.parts]

        # Prefer same rover subtree.
        for rover in ("m2020", "msl", "mer"):
            if rover in mask_parts and rover in image_parts:
                score += 5

        # Prefer train↔train-like and test↔test-like directory affinity.
        if "test" in mask_parts and "test" in image_parts:
            score += 4
        if "train" in mask_parts and "test" not in image_parts:
            score += 2

        # Camera/subset affinity.
        for token in ("ncam", "mcam", "edr", "eff", "mxy"):
            if token in mask_parts and token in image_parts:
                score += 3

        return score

    ranked = sorted(((score_candidate(path), path) for path in candidates), reverse=True)
    best_score = ranked[0][0]
    best_paths = [path for score, path in ranked if score == best_score]

    if len(best_paths) > 1:
        # Common in merged archives: the same image exists as .jpg and .jpeg.
        # Prefer a stable extension ordering before declaring ambiguity.
        extension_priority = {".jpg": 3, ".jpeg": 2, ".tif": 1, ".tiff": 0}
        best_by_ext = max(extension_priority.get(path.suffix.lower(), -1) for path in best_paths)
        best_paths = [
            path
            for path in best_paths
            if extension_priority.get(path.suffix.lower(), -1) == best_by_ext
        ]

    if len(best_paths) != 1:
        candidate_list = "\n".join(f"  - {path}" for path in sorted(candidates))
        raise RuntimeError(
            "Ambiguous image match for mask path with duplicate image stems. "
            f"Mask: {mask_path}\nCandidates:\n{candidate_list}"
        )

    return best_paths[0]


def _find_source_image_for_mask(mask_path: Path, images_by_stem: Dict[str, List[Path]]) -> Optional[Path]:
    """Find the source image path for a given mask path."""
    segments = mask_path.stem.split("_")
    for cutoff in range(len(segments), 0, -1):
        candidate_stem = "_".join(segments[:cutoff])
        candidate_images = images_by_stem.get(candidate_stem)
        if candidate_images:
            return _select_image_candidate_for_mask(mask_path, candidate_images)
    return None
