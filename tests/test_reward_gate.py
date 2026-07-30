from __future__ import annotations

import unittest

from eviback.prompts import INSUFFICIENT_ANSWER
from eviback.rewards.eviback_reward import EviBackReward, RewardConfig, Trajectory
from eviback.teacher.client import MockTeacherClient
from eviback.teacher.strategy import EvidenceConstrainedTeacher


def xml(status: str, answer: str) -> str:
    return f"<reason>Evidence check.</reason><status>{status}</status><answer>{answer}</answer>"


def record(index: int, answer: str, *, malformed: bool = False, evidence: bool = True) -> Trajectory:
    output = "broken" if malformed else f"<reason>Done.</reason><answer>{answer}</answer>"
    steps = (
        {
            "sub_query": "capital France",
            "docs": [{"contents": "Paris is the capital of France."}],
        },
    ) if evidence else ()
    return Trajectory(
        group_id="group",
        question="What is the capital of France?",
        output=output,
        reference_answers=("Paris",),
        evidence_steps=steps,
        index=str(index),
    )


class RewardGateTest(unittest.TestCase):
    def make_reward(self, responses):
        client = MockTeacherClient(responses)
        return client, EviBackReward(
            EvidenceConstrainedTeacher(client), RewardConfig(group_size=8)
        )

    def test_all_zero_calls_teacher_and_uses_fallback_scale(self):
        client, reward = self.make_reward(
            [value for _ in range(8) for value in (xml("supported_answer", "Paris"), xml("supported_answer", "Paris"))]
        )
        results = reward.compute([record(index, "London") for index in range(8)])
        self.assertEqual(client.call_count, 16)
        self.assertTrue(all(result.score == 0.2 for result in results))
        self.assertEqual({result.metadata["advantage_postnorm_scale"] for result in results}, {0.1})

    def test_mixed_group_keeps_actor_reward_and_never_calls_teacher(self):
        client, reward = self.make_reward([])
        results = reward.compute([record(0, "Paris"), *[record(index, "London") for index in range(1, 8)]])
        self.assertEqual(client.call_count, 0)
        self.assertEqual([result.score for result in results], [1.0] + [0.0] * 7)
        self.assertEqual({result.metadata["advantage_postnorm_scale"] for result in results}, {1.0})

    def test_all_one_never_calls_teacher(self):
        client, reward = self.make_reward([])
        results = reward.compute([record(index, "Paris") for index in range(8)])
        self.assertEqual(client.call_count, 0)
        self.assertTrue(all(result.score == 1.0 for result in results))

    def test_malformed_actor_cannot_receive_answer_bonus(self):
        client, reward = self.make_reward([xml("insufficient_evidence", INSUFFICIENT_ANSWER)] * 8)
        results = reward.compute([record(index, "", malformed=True) for index in range(8)])
        self.assertEqual(client.call_count, 8)
        self.assertTrue(all(result.score == 0.0 for result in results))
        self.assertTrue(all(not result.metadata["teacher_gold_token_f1_bonus_eligible"] for result in results))

    def test_no_evidence_skips_teacher_but_keeps_group_scale(self):
        client, reward = self.make_reward([])
        results = reward.compute([record(index, "London", evidence=False) for index in range(8)])
        self.assertEqual(client.call_count, 0)
        self.assertEqual({result.metadata["advantage_postnorm_scale"] for result in results}, {0.1})


if __name__ == "__main__":
    unittest.main()