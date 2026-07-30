from __future__ import annotations

import unittest

import numpy as np

from eviback.training.postnorm import compute_group_normalized_advantages


class GroupPostnormTest(unittest.TestCase):
    def test_scale_applies_after_normalization(self):
        pre = compute_group_normalized_advantages([0.0, 1.0, 0.0, 1.0], ["a", "a", "b", "b"])
        post = compute_group_normalized_advantages(
            [0.0, 1.0, 0.0, 1.0],
            ["a", "a", "b", "b"],
            scales=[1.0, 1.0, 0.1, 0.1],
        )
        np.testing.assert_allclose(post[:2], pre[:2])
        np.testing.assert_allclose(post[2:], pre[2:] * 0.1)

    def test_rejects_inconsistent_group_scale(self):
        with self.assertRaises(ValueError):
            compute_group_normalized_advantages(
                [0.0, 1.0], ["group", "group"], scales=[0.1, 1.0]
            )

    def test_rejects_nonpositive_scale(self):
        with self.assertRaises(ValueError):
            compute_group_normalized_advantages([0.0, 1.0], ["a", "a"], scales=[0.0, 0.0])


if __name__ == "__main__":
    unittest.main()