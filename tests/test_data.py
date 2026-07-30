from __future__ import annotations

import unittest

from eviback.data import stable_sample


class DataTest(unittest.TestCase):
    def test_stable_sample_deduplicates_questions(self):
        rows = [
            {"id": "2", "question": "Same question", "answers": ["a"]},
            {"id": "1", "question": " Same   question ", "answers": ["a"]},
            {"id": "3", "question": "Other", "answers": ["b"]},
        ]
        first = stable_sample(rows, "nq", 2, 42)
        second = stable_sample(list(reversed(rows)), "nq", 2, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()