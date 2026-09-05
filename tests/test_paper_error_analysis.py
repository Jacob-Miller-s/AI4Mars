import json
import tempfile
import unittest
from pathlib import Path

from ai4mars.paper_error_analysis import regenerate_artifacts, summarize_bedrock_big_rock


class PaperErrorAnalysisTests(unittest.TestCase):
    def test_regenerates_artifacts_without_model_or_dataset(self) -> None:
        matrix = [[8, 0, 0, 0], [0, 6, 0, 2], [0, 0, 4, 0], [0, 3, 0, 5]]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "expert_evaluation.json"
            artifact.write_text(json.dumps({"splits": {"expert_min1": {"confusion_matrix": matrix}}}), encoding="utf-8")
            summaries = regenerate_artifacts(artifact, root / "plots")

            self.assertTrue((root / "plots/expert_min1_confusion_matrix_raw.png").is_file())
            self.assertTrue((root / "plots/expert_min1_confusion_matrix_normalized.csv").is_file())
            self.assertEqual(summaries[0]["ground_truth_bedrock_predicted_big_rock_pixels"], 2)
            self.assertEqual(summaries[0]["ground_truth_big_rock_predicted_bedrock_pixels"], 3)

    def test_bedrock_big_rock_rates_use_ground_truth_row_support(self) -> None:
        summary = summarize_bedrock_big_rock(__import__("numpy").array([[1, 0, 0, 0], [0, 6, 0, 2], [0, 0, 1, 0], [0, 3, 0, 5]]), "expert_min1")
        self.assertEqual(summary["ground_truth_bedrock_predicted_big_rock_rate"], 0.25)
        self.assertEqual(summary["ground_truth_big_rock_predicted_bedrock_rate"], 0.375)


if __name__ == "__main__":
    unittest.main()