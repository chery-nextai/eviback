from __future__ import annotations

import unittest

from eviback.evaluation import align_predictions, paired_bootstrap, score_aligned, summarize
from eviback.metrics import trajectory_metrics


class MetricsTest(unittest.TestCase):
    def test_alignment_and_metrics(self):
        references = [
            {"data_source": "nq", "index": 1, "question": "Capital?", "answers": ["Paris"]}
        ]
        predictions = [
            {
                "data_source": "nq",
                "index": 1,
                "question": "Capital?",
                "final_answer": "Paris",
                "status": "answered",
                "queries": ["q", "q"],
                "search_count": 2,
            }
        ]
        rows = score_aligned(align_predictions(predictions, references))
        summary = summarize(rows)
        self.assertEqual(summary["overall"]["legacy_em"], 1.0)
        self.assertEqual(summary["overall"]["duplicate_query_rate"], 1.0)

    def test_alignment_rejects_question_mismatch(self):
        with self.assertRaises(ValueError):
            align_predictions(
                [{"data_source": "nq", "index": 1, "question": "A"}],
                [{"data_source": "nq", "index": 1, "question": "B"}],
            )

    def test_bootstrap_is_paired_and_deterministic(self):
        first = paired_bootstrap([0, 0, 1, 1], [1, 0, 1, 1], samples=1000, seed=42)
        second = paired_bootstrap([0, 0, 1, 1], [1, 0, 1, 1], samples=1000, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_minus_baseline"], 0.25)

    def test_duplicate_and_max_turn_trajectory_metrics(self):
        text = (
            '<reason>a</reason><tool_call>{"name":"search","arguments":{"query":"same"}}</tool_call>'
            "\nuser\n<tool_response>x</tool_response>\nassistant\n"
            '<reason>b</reason><tool_call>{"name":"search","arguments":{"query":"same"}}</tool_call>'
            "\nuser\n<tool_response>x</tool_response>\nassistant\n"
            "<reason>c</reason><answer>answer</answer>"
        )
        metrics = trajectory_metrics(text, max_turns=3)
        self.assertEqual(metrics["duplicate_query"], 1.0)
        self.assertEqual(metrics["maximum_turn"], 1.0)


if __name__ == "__main__":
    unittest.main()