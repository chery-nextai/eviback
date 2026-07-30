"""BatchRewardManager entry point for an installed VERL trainer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from eviback.rewards.eviback_reward import EviBackReward, RewardConfig, Trajectory
from eviback.teacher.client import OpenAICompatibleTeacherClient
from eviback.teacher.strategy import EvidenceConstrainedTeacher, STRATEGY_ID


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


def _references(ground_truth: Any) -> tuple[str, ...]:
    if isinstance(ground_truth, Mapping):
        raw = ground_truth.get("target")
        if raw is None:
            raw = ground_truth.get("answers")
        if raw is None:
            raw = ground_truth.get("answer")
    else:
        raw = ground_truth
    return tuple(str(value) for value in _values(raw) if str(value).strip())


def _evidence(extra: Mapping[str, Any], visible_top_m: int) -> tuple[dict[str, Any], ...]:
    steps: list[dict[str, Any]] = []
    for turn, detail in enumerate(_values(extra.get("tool_call_details")), start=1):
        if not isinstance(detail, Mapping):
            continue
        docs: list[dict[str, Any]] = []
        for key in ("top_5_documents", "rank_top5_docs", "rank_top50_docs", "recall_top50_docs"):
            raw_docs = detail.get(key)
            if isinstance(raw_docs, list):
                docs = [dict(value) for value in raw_docs[:visible_top_m] if isinstance(value, Mapping)]
                break
        steps.append(
            {
                "turn": turn,
                "sub_query": str(detail.get("sub_query") or ""),
                "docs": docs,
            }
        )
    return tuple(steps)


def compute_eviback_reward_batch(
    data_sources: Sequence[Any],
    solution_strs: Sequence[Any],
    ground_truths: Sequence[Any],
    extra_infos: Sequence[Mapping[str, Any]],
    *,
    teacher_request: Mapping[str, Any],
    n_samples_per_prompt: int = 8,
    visible_top_m: int = 5,
    partial_reward: float = 0.1,
    gold_token_f1_bonus: float = 0.1,
    teacher_fallback_scale: float = 0.1,
    teacher_strategy_id: str = STRATEGY_ID,
    **_: Any,
) -> list[dict[str, Any]]:
    """Translate VERL batch fields into the public EviBack reward contract."""

    item_count = len(solution_strs)
    if not (
        len(data_sources) == len(ground_truths) == len(extra_infos) == item_count
    ):
        raise ValueError("VERL reward batch inputs must have the same length")
    if item_count == 0:
        return []
    if teacher_strategy_id != STRATEGY_ID:
        raise ValueError(f"EviBack requires teacher_strategy_id={STRATEGY_ID}")
    client = OpenAICompatibleTeacherClient(
        endpoint=str(teacher_request.get("endpoint") or ""),
        model=str(teacher_request.get("model") or ""),
        timeout_seconds=float(teacher_request.get("timeout_seconds", 180)),
        temperature=float(teacher_request.get("temperature", 0.0)),
        top_p=float(teacher_request.get("top_p", 1.0)),
        max_tokens=int(teacher_request.get("max_tokens", 512)),
    )
    reward = EviBackReward(
        EvidenceConstrainedTeacher(client, visible_top_m=visible_top_m),
        RewardConfig(
            group_size=n_samples_per_prompt,
            partial_reward=partial_reward,
            gold_token_f1_bonus=gold_token_f1_bonus,
            teacher_fallback_scale=teacher_fallback_scale,
        ),
    )
    records = [
        Trajectory(
            group_id=str(extra_infos[index].get("uid") or ""),
            question=str(
                extra_infos[index].get("question")
                or extra_infos[index].get("initial_query")
                or ""
            ),
            output=str(solution_strs[index]),
            reference_answers=_references(ground_truths[index]),
            evidence_steps=_evidence(extra_infos[index], visible_top_m),
            data_source=str(data_sources[index]),
            index=str(extra_infos[index].get("index") or ""),
        )
        for index in range(item_count)
    ]
    return [result.metadata for result in reward.compute(records)]