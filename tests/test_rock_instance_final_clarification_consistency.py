import json
import tempfile
import unittest
from pathlib import Path

from src.rock_instance.final_clarification_consistency import _completed_target, write_analysis_outputs


class FinalClarificationConsistencyTests(unittest.TestCase):
    def test_completed_target_allows_component_in_prior_multi_target_scope(self) -> None:
        state = {
            "schema_version": "prior-schema",
            "review_scope": {"target_ids": ["target-1", "target-2"]},
            "targets": [
                {"target_id": "target-1", "review_status": "redrawn", "object_identity_fixed": "accepted", "identity_escalation": False},
                {"target_id": "target-2", "review_status": "redrawn", "object_identity_fixed": "accepted", "identity_escalation": False},
            ],
        }
        self.assertEqual(_completed_target(state, schema_version="prior-schema", target_id="target-2", isolated_scope=False)["target_id"], "target-2")
        with self.assertRaises(ValueError):
            _completed_target(state, schema_version="prior-schema", target_id="target-2", isolated_scope=True)

    def test_write_analysis_outputs_writes_json_and_markdown(self) -> None:
        report = {
            "comparisons": [{"comparison": "v2.2.1_to_v2.2", "v221_area_pixels": 10, "previous_area_pixels": 8, "v221_to_previous_area_ratio": 1.25, "mask_iou": 0.5}],
            "qualitative_review_required": "Assess visible extent.",
            "CALIBRATION_PROTOCOL_RECOMMENDATION": "CLARIFY_AGAIN",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            markdown_path = Path(temporary_directory) / "report.md"
            write_analysis_outputs(report, output_dir, markdown_path)
            self.assertEqual(json.loads((output_dir / "final_clarification_consistency.json").read_text(encoding="utf-8")), report)
            self.assertIn("v2.2.1_to_v2.2", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()