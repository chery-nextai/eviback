from __future__ import annotations

import unittest

from eviback.parsers import EvidenceStatus, parse_actor_trajectory, parse_teacher_output
from eviback.prompts import INSUFFICIENT_ANSWER


class TeacherParserTest(unittest.TestCase):
    def test_parses_strict_xml(self):
        result = parse_teacher_output(
            "<reason>The passage states it.</reason>"
            "<status>supported_answer</status><answer>Paris</answer>"
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.status, EvidenceStatus.SUPPORTED)
        self.assertEqual(result.answer, "Paris")

    def test_rejects_truncation_extra_text_and_invalid_status(self):
        cases = [
            "<reason>x</reason><status>supported_answer</status><answer>Paris",
            "prefix<reason>x</reason><status>supported_answer</status><answer>Paris</answer>",
            "<reason>x</reason><status>maybe</status><answer>Paris</answer>",
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertFalse(parse_teacher_output(value).valid)

    def test_rejects_duplicate_xml_tags(self):
        result = parse_teacher_output(
            "<reason>one<reason>two</reason></reason>"
            "<status>supported_answer</status><answer>Paris</answer>"
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.error_code, "invalid_reason_tag_count")

    def test_enforces_status_answer_contract(self):
        bad = parse_teacher_output(
            "<reason>Missing bridge.</reason><status>insufficient_evidence</status><answer>Paris</answer>"
        )
        self.assertEqual(bad.error_code, "status_answer_contract_error")
        good = parse_teacher_output(
            f"<reason>Missing bridge.</reason><status>insufficient_evidence</status><answer>{INSUFFICIENT_ANSWER}</answer>"
        )
        self.assertTrue(good.valid)


class ActorParserTest(unittest.TestCase):
    def test_parses_search_then_answer(self):
        trajectory = (
            '<reason>Search.</reason><tool_call>{"name":"search","arguments":{"query":"capital France"}}</tool_call>'
            "\nuser\n<tool_response>Paris is the capital.</tool_response>\nassistant\n"
            "<reason>Answer.</reason><answer>Paris</answer>"
        )
        result = parse_actor_trajectory(trajectory)
        self.assertTrue(result.valid)
        self.assertEqual(result.answer, "Paris")
        self.assertEqual(result.queries, ("capital France",))

    def test_malformed_actor_is_conservative(self):
        result = parse_actor_trajectory(
            '<reason>x</reason><tool_call>{"name":"search","arguments":{}}</tool_call>'
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.error_code, "invalid_tool_call")


if __name__ == "__main__":
    unittest.main()