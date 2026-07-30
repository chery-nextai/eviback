#!/usr/bin/env python3
"""GPU-free EviBack reward smoke test with a deterministic Teacher."""

from __future__ import annotations

import argparse
import json

from eviback.rewards.eviback_reward import EviBackReward, RewardConfig, Trajectory
from eviback.teacher.client import MockTeacherClient
from eviback.teacher.strategy import EvidenceConstrainedTeacher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher/default.yaml")
    args = parser.parse_args()
    del args

    def response(messages, call_index):
        del call_index
        is_stage_b = "Reference answer" in messages[-1]["content"]
        reason = "Evidence explicitly states the requested capital."
        answer = "Paris" if is_stage_b else "Paris"
        return f"<reason>{reason}</reason><status>supported_answer</status><answer>{answer}</answer>"

    client = MockTeacherClient(response)
    teacher = EvidenceConstrainedTeacher(client)
    reward = EviBackReward(teacher, RewardConfig(group_size=8))
    evidence = (
        {
            "sub_query": "capital of France",
            "docs": [{"title": "France", "contents": "Paris is the capital of France."}],
        },
    )
    records = [
        Trajectory(
            group_id="smoke-question",
            question="What is the capital of France?",
            output=f"<reason>I found an answer.</reason><answer>Wrong-{index}</answer>",
            reference_answers=("Paris",),
            evidence_steps=evidence,
        )
        for index in range(8)
    ]
    results = reward.compute(records)
    report = {
        "count": len(results),
        "scores": [result.score for result in results],
        "teacher_calls": client.call_count,
        "all_zero_fallback": all(result.metadata["group_all_em_zero"] for result in results),
        "group_scales": sorted({result.metadata["advantage_postnorm_scale"] for result in results}),
    }
    print(json.dumps(report, indent=2))
    if report != {
        "count": 8,
        "scores": [0.2] * 8,
        "teacher_calls": 16,
        "all_zero_fallback": True,
        "group_scales": [0.1],
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())