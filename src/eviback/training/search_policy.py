"""Framework-neutral reward adapter used from a VERL reward manager."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eviback.rewards.eviback_reward import EviBackReward, RewardResult, Trajectory


def reward_batch(
    reward: EviBackReward, records: Sequence[Trajectory | dict[str, Any]]
) -> tuple[list[float], dict[str, list[Any]]]:
    results: list[RewardResult] = reward.compute(records)
    keys = sorted({key for result in results for key in result.metadata})
    metadata = {
        key: [result.metadata.get(key) for result in results]
        for key in keys
    }
    return [result.score for result in results], metadata