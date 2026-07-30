"""Answer and trajectory metrics used by EviBack evaluation and rewards."""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from eviback.parsers import parse_actor_trajectory


def normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(prediction: Any, answers: Sequence[Any]) -> float:
    prediction_normalized = normalize_answer(prediction)
    if not prediction_normalized:
        return 0.0
    return float(any(prediction_normalized == normalize_answer(answer) for answer in answers))


def token_f1_pair(prediction: Any, answer: Any) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not prediction_tokens or not answer_tokens:
        return 0.0
    common = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: Any, answers: Sequence[Any]) -> float:
    return max((token_f1_pair(prediction, answer) for answer in answers), default=0.0)


def trajectory_metrics(trajectory: Any, *, max_turns: int | None = None) -> dict[str, float]:
    parsed = parse_actor_trajectory(trajectory, max_turns=max_turns)
    return {
        "valid_answer": float(parsed.valid and bool(parsed.answer)),
        "search_calls": float(parsed.search_count),
        "duplicate_query": float(parsed.duplicate_query_count > 0),
        "duplicate_query_count": float(parsed.duplicate_query_count),
        "maximum_turn": float(parsed.status == "max_turns"),
        "turn_count": float(parsed.assistant_turn_count),
    }


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    materialized = list(rows)
    keys = sorted({key for row in materialized for key in row})
    return {
        key: sum(float(row.get(key, 0.0)) for row in materialized) / len(materialized)
        if materialized
        else 0.0
        for key in keys
    }


def macro_by_source(
    rows: Iterable[dict[str, Any]], metric_names: Sequence[str]
) -> dict[str, float]:
    per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_source[str(row.get("data_source") or "unknown")].append(row)
    source_means = {
        source: {
            metric: sum(float(row.get(metric, 0.0)) for row in source_rows) / len(source_rows)
            for metric in metric_names
        }
        for source, source_rows in per_source.items()
        if source_rows
    }
    return {
        metric: sum(values[metric] for values in source_means.values()) / len(source_means)
        if source_means
        else 0.0
        for metric in metric_names
    }