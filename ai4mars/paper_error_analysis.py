"""Regenerate expert confusion artifacts from saved evaluation JSON without inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai4mars.paper_reproduction import CLASS_NAMES


def _as_square_matrix(raw: Any, *, split_name: str) -> np.ndarray:
    matrix = np.asarray(raw, dtype=np.int64)
    expected = (len(CLASS_NAMES), len(CLASS_NAMES))
    if matrix.shape != expected or (matrix < 0).any():
        raise ValueError(f"{split_name} confusion_matrix must be a non-negative {expected} matrix.")
    return matrix


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    support = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth \\ predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])


def write_confusion_figure(path: Path, matrix: np.ndarray, *, split_name: str, normalized: bool) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0 if normalized else None, vmax=1.0 if normalized else None)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Ground-truth class")
    title_suffix = "row-normalized (diagonal = recall)" if normalized else "raw pixel counts"
    axis.set_title(f"{split_name}: confusion matrix ({title_suffix})")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            label = f"{value:.3f}" if normalized else f"{int(value):,}"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8, color="white" if normalized and value >= 0.5 else "black")
    figure.colorbar(image, ax=axis, label="Row-normalized frequency" if normalized else "Pixel count")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def summarize_bedrock_big_rock(matrix: np.ndarray, split_name: str) -> dict[str, Any]:
    bedrock_index, big_rock_index = 1, 3
    bedrock_support = int(matrix[bedrock_index].sum())
    big_rock_support = int(matrix[big_rock_index].sum())
    return {
        "split": split_name,
        "ground_truth_bedrock_predicted_big_rock_pixels": int(matrix[bedrock_index, big_rock_index]),
        "ground_truth_bedrock_predicted_big_rock_rate": matrix[bedrock_index, big_rock_index] / bedrock_support if bedrock_support else 0.0,
        "ground_truth_big_rock_predicted_bedrock_pixels": int(matrix[big_rock_index, bedrock_index]),
        "ground_truth_big_rock_predicted_bedrock_rate": matrix[big_rock_index, bedrock_index] / big_rock_support if big_rock_support else 0.0,
    }


def regenerate_artifacts(evaluation_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Regenerate CSV/PNG artifacts from a saved expert evaluation report only."""
    report = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    splits = report.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Evaluation artifact must contain a non-empty splits mapping.")
    summaries: list[dict[str, Any]] = []
    for split_name in sorted(splits):
        matrix = _as_square_matrix(splits[split_name].get("confusion_matrix"), split_name=split_name)
        normalized = _row_normalize(matrix)
        write_confusion_csv(output_dir / f"{split_name}_confusion_matrix_raw.csv", matrix)
        write_confusion_csv(output_dir / f"{split_name}_confusion_matrix_normalized.csv", normalized)
        write_confusion_figure(output_dir / f"{split_name}_confusion_matrix_raw.png", matrix, split_name=split_name, normalized=False)
        write_confusion_figure(output_dir / f"{split_name}_confusion_matrix_normalized.png", normalized, split_name=split_name, normalized=True)
        summaries.append(summarize_bedrock_big_rock(matrix, split_name))
    with (output_dir / "bedrock_big_rock_confusion_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = regenerate_artifacts(args.evaluation_artifact, args.output_dir)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()