"""Actor answer reward and trajectory diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eviback.metrics import exact_match, token_f1
from eviback.parsers import parse_actor_trajectory


def compute_actor_reward(
    output: Any, reference_answers: Sequence[Any], *, max_turns: int | None = None
) -> dict[str, Any]:
    parsed = parse_actor_trajectory(output, max_turns=max_turns)
    em = exact_match(parsed.answer, reference_answers) if parsed.valid else 0.0
    f1 = token_f1(parsed.answer, reference_answers) if parsed.valid else 0.0
    return {
        "score": em,
        "actor_answer": parsed.answer,
        "actor_answer_parse_status": "parsed" if parsed.valid else parsed.error_code,
        "actor_valid": parsed.valid,
        "actor_em": em,
        "actor_token_f1": f1,
        "search_count": parsed.search_count,
        "duplicate_query_count": parsed.duplicate_query_count,
        "max_turn_reached": parsed.status == "max_turns",
        "turn_count": parsed.assistant_turn_count,
        "trajectory_status": parsed.status,
    }