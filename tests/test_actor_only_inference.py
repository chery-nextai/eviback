from __future__ import annotations

import unittest

from eviback.inference import run_actor_inference
from eviback.retrieval.client import RetrievedDocument


class FakeActor:
    def __init__(self):
        self.responses = iter(
            [
                '<reason>Search.</reason><tool_call>{"name":"search","arguments":{"query":"capital France"}}</tool_call>',
                "<reason>The evidence answers it.</reason><answer>Paris</answer>",
            ]
        )

    def complete(self, messages):
        return next(self.responses)


class FakeRetriever:
    def retrieve(self, query, *, top_k=50):
        self.query = query
        self.top_k = top_k
        return [RetrievedDocument("1", "France", "Paris is the capital.", 1.0, 1)]


class ActorOnlyInferenceTest(unittest.TestCase):
    def test_actor_only_trace(self):
        result = run_actor_inference(
            "What is the capital of France?", actor=FakeActor(), retriever=FakeRetriever()
        )
        self.assertEqual(result["final_answer"], "Paris")
        self.assertFalse(result["teacher_used"])
        self.assertFalse(result["reference_answer_used"])
        self.assertEqual(result["queries"], ["capital France"])


if __name__ == "__main__":
    unittest.main()