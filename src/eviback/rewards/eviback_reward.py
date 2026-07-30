"""Actor-first EviBack reward with all-zero Teacher fallback."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from eviback.metrics import token_f1
from eviback.parsers import EvidenceStatus
from eviback.rewards.actor_reward import compute_actor_reward
from eviback.rewards.reward_metadata import (
    ADVANTAGE_SCALE_KEY,
    ADVANTAGE_SCALE_VERSION,
    BONUS_ELIGIBILITY_VERSION,
    REWARD_VERSION,
    empty_teacher_metadata,
)
from eviback.teacher.strategy import EvidenceConstrainedTeacher


@dataclass(frozen=True)
class RewardConfig:
    group_size: int | None = 8
    partial_reward: float = 0.1
    gold_token_f1_bonus: float = 0.1
    teacher_fallback_scale: float = 0.1
    max_turns: int | None = None

    def __post_init__(self) -> None:
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive or None")
        if self.partial_reward < 0:
            raise ValueError("partial_reward must be non-negative")
        if self.gold_token_f1_bonus < 0:
            raise ValueError("gold_token_f1_bonus must be non-negative")
        if not 0 < self.teacher_fallback_scale <= 1:
            raise ValueError("teacher_fallback_scale must be in (0, 1]")


@dataclass(frozen=True)
class Trajectory:
    group_id: str
    question: str
    output: str
    reference_answers: tuple[str, ...]
    evidence_steps: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    data_source: str = ""
    index: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Trajectory":
        references = value.get("reference_answers")
        if references is None:
            references = value.get("answers") or value.get("gold_answers") or []
        if isinstance(references, str):
            references = [references]
        return cls(
            group_id=str(value.get("group_id") or value.get("uid") or ""),
            question=str(value.get("question") or ""),
            output=str(value.get("output") or value.get("trajectory") or ""),
            reference_answers=tuple(str(item) for item in references if str(item).strip()),
            evidence_steps=tuple(value.get("evidence_steps") or value.get("evidence") or ()),
            data_source=str(value.get("data_source") or ""),
            index=str(value.get("index") or ""),
        )


@dataclass(frozen=True)
class RewardResult:
    score: float
    metadata: dict[str, Any]


class EviBackReward:
    """Compute group rewards while enforcing the Actor-first gate."""

    def __init__(self, teacher: EvidenceConstrainedTeacher, config: RewardConfig | None = None) -> None:
        self.teacher = teacher
        self.config = config or RewardConfig()

    def compute(self, trajectories: Sequence[Trajectory | dict[str, Any]]) -> list[RewardResult]:
        items = [
            item if isinstance(item, Trajectory) else Trajectory.from_mapping(item)
            for item in trajectories
        ]
        if not items:
            return []
        groups: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(items):
            if not item.group_id:
                raise ValueError(f"trajectory {index} has no group_id")
            if not item.reference_answers:
                raise ValueError(f"trajectory {index} has no reference answers")
            groups[item.group_id].append(index)
        if self.config.group_size is not None:
            invalid = {
                group_id: len(indices)
                for group_id, indices in groups.items()
                if len(indices) != self.config.group_size
            }
            if invalid:
                raise ValueError(
                    f"each group must have {self.config.group_size} trajectories; got {invalid}"
                )

        actor_details = [
            compute_actor_reward(
                item.output, item.reference_answers, max_turns=self.config.max_turns
            )
            for item in items
        ]
        all_zero = {
            group_id: not any(actor_details[index]["actor_em"] >= 1.0 for index in indices)
            for group_id, indices in groups.items()
        }

        results: list[RewardResult] = []
        for index, item in enumerate(items):
            actor = actor_details[index]
            fallback = all_zero[item.group_id]
            metadata: dict[str, Any] = {
                **actor,
                **empty_teacher_metadata(),
                "reward_type": REWARD_VERSION,
                "group_uid": item.group_id,
                "group_size": len(groups[item.group_id]),
                "group_all_em_zero": fallback,
                "partial_reward_applied": fallback,
                "advantage_source": "teacher_fallback" if fallback else "actor_em",
                ADVANTAGE_SCALE_KEY: (
                    self.config.teacher_fallback_scale if fallback else 1.0
                ),
                "advantage_postnorm_scale_version": ADVANTAGE_SCALE_VERSION,
                "teacher_gold_token_f1_bonus_eligibility_version": BONUS_ELIGIBILITY_VERSION,
                "data_source": item.data_source,
                "index": item.index,
            }
            if not fallback:
                metadata["teacher_skip_reason"] = "group_has_positive_actor_em"
                metadata.update(
                    {
                        "base_reward": float(actor["actor_em"]),
                        "teacher_gold_token_f1": 0.0,
                        "teacher_gold_token_f1_bonus": 0.0,
                        "teacher_gold_token_f1_bonus_eligible": False,
                        "score": float(actor["actor_em"]),
                    }
                )
                results.append(RewardResult(float(actor["actor_em"]), metadata))
                continue
            if not item.evidence_steps:
                metadata["teacher_skip_reason"] = "no_search_evidence"
                metadata.update(
                    {
                        "base_reward": 0.0,
                        "teacher_gold_token_f1": 0.0,
                        "teacher_gold_token_f1_bonus": 0.0,
                        "teacher_gold_token_f1_bonus_eligible": False,
                        "score": 0.0,
                    }
                )
                results.append(RewardResult(0.0, metadata))
                continue

            decision = self.teacher.judge(
                question=item.question,
                evidence_steps=item.evidence_steps,
                reference_answers=item.reference_answers,
            )
            metadata.update(decision.metadata)
            metadata["teacher_called"] = True
            metadata["teacher_skip_reason"] = ""
            supported = bool(
                decision.result.valid
                and decision.result.status
                in {EvidenceStatus.SUPPORTED, EvidenceStatus.AMBIGUOUS}
            )
            base_reward = self.config.partial_reward * float(supported)
            eligible = bool(supported and actor["actor_valid"] and decision.result.answer)
            teacher_f1 = (
                token_f1(decision.result.answer, item.reference_answers) if eligible else 0.0
            )
            bonus = self.config.gold_token_f1_bonus * teacher_f1
            score = base_reward + bonus
            metadata.update(
                {
                    "teacher_status_reward": float(supported),
                    "base_reward": base_reward,
                    "teacher_gold_token_f1": teacher_f1,
                    "teacher_gold_token_f1_bonus": bonus,
                    "teacher_gold_token_f1_bonus_applied": bonus > 0,
                    "teacher_gold_token_f1_bonus_eligible": eligible,
                    "score": score,
                }
            )
            results.append(RewardResult(score, metadata))
        return results


def compute_eviback_rewards(
    trajectories: Sequence[Trajectory | dict[str, Any]],
    *,
    teacher: EvidenceConstrainedTeacher,
    config: RewardConfig | None = None,
) -> list[dict[str, Any]]:
    """Framework-friendly wrapper returning one metadata mapping per response."""

    return [result.metadata for result in EviBackReward(teacher, config).compute(trajectories)]