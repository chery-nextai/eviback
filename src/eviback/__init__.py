"""Public EviBack interfaces."""

from eviback.metrics import exact_match, normalize_answer, token_f1
from eviback.rewards.eviback_reward import EviBackReward, RewardConfig, Trajectory
from eviback.teacher.strategy import EvidenceConstrainedTeacher

__all__ = [
    "EviBackReward",
    "EvidenceConstrainedTeacher",
    "RewardConfig",
    "Trajectory",
    "exact_match",
    "normalize_answer",
    "token_f1",
]

__version__ = "0.1.0rc1"