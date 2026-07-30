from __future__ import annotations

import unittest

from eviback.prompts import INSUFFICIENT_ANSWER, build_stage_a_messages, build_stage_b_messages
from eviback.teacher.client import MockTeacherClient
from eviback.teacher.merge import find_literal_reference
from eviback.teacher.strategy import EvidenceConstrainedTeacher


def xml(status: str, answer: str) -> str:
    return f"<reason>Evidence check.</reason><status>{status}</status><answer>{answer}</answer>"


EVIDENCE = [
    {
        "sub_query": "capital France",
        "docs": [{"title": "France", "contents": "Paris is the capital of France."}],
    }
]


class TeacherContractTest(unittest.TestCase):
    def test_stage_a_insufficient_never_calls_stage_b(self):
        client = MockTeacherClient([xml("insufficient_evidence", INSUFFICIENT_ANSWER)])
        decision = EvidenceConstrainedTeacher(client).judge(
            question="What is the capital?",
            evidence_steps=EVIDENCE,
            reference_answers=["Paris"],
        )
        self.assertEqual(client.call_count, 1)
        self.assertFalse(decision.metadata["teacher_stage_b_called"])
        self.assertTrue(decision.metadata["teacher_i_boundary_preserved"])

    def test_stage_b_failure_falls_back_to_stage_a(self):
        client = MockTeacherClient([xml("supported_answer", "Paris"), "truncated"])
        decision = EvidenceConstrainedTeacher(client).judge(
            question="What is the capital of France?",
            evidence_steps=EVIDENCE,
            reference_answers=["Paris"],
        )
        self.assertEqual(decision.result.answer, "Paris")
        self.assertEqual(decision.merge.selected_stage, "stage_a")
        self.assertFalse(decision.merge.stage_b_used)

    def test_stage_b_supported_can_be_selected(self):
        client = MockTeacherClient(
            [xml("ambiguous_evidence", INSUFFICIENT_ANSWER), xml("supported_answer", "Paris")]
        )
        decision = EvidenceConstrainedTeacher(client).judge(
            question="What is the capital of France?",
            evidence_steps=EVIDENCE,
            reference_answers=["Paris"],
        )
        self.assertEqual(decision.merge.selected_stage, "stage_b")
        self.assertEqual(decision.result.answer, "Paris")

    def test_reference_is_separate_from_evidence(self):
        stage_a = build_stage_a_messages("Question", EVIDENCE)
        stage_b = build_stage_b_messages("Question", ["Secret Gold"], EVIDENCE)
        self.assertNotIn("Secret Gold", stage_a[-1]["content"])
        self.assertIn("candidate only, not evidence", stage_b[-1]["content"])
        self.assertEqual(find_literal_reference(EVIDENCE, ["Secret Gold"]), "")
        self.assertEqual(find_literal_reference(EVIDENCE, ["Paris"]), "Paris")


if __name__ == "__main__":
    unittest.main()