import unittest

from src.rock_instance.clarification_analysis import recommendation_for_resolution_rows


class ClarificationAnalysisTests(unittest.TestCase):
    def test_recommendation_requires_every_targeted_discrepancy_to_be_resolved(self) -> None:
        self.assertEqual(
            recommendation_for_resolution_rows([
                {"fully_resolved_by_v21_rule": True},
                {"fully_resolved_by_v21_rule": True},
            ]),
            "FREEZE",
        )
        self.assertEqual(
            recommendation_for_resolution_rows([
                {"fully_resolved_by_v21_rule": True},
                {"fully_resolved_by_v21_rule": False},
            ]),
            "CLARIFY_AGAIN",
        )
        self.assertEqual(recommendation_for_resolution_rows([]), "CLARIFY_AGAIN")


if __name__ == "__main__":
    unittest.main()