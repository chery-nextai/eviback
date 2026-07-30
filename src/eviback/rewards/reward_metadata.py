"""Stable reward metadata contract."""

from __future__ import annotations

from typing import Any

REWARD_VERSION = "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2"
ADVANTAGE_SCALE_KEY = "advantage_postnorm_scale"
ADVANTAGE_SCALE_VERSION = "teacher_fallback_v1"
BONUS_ELIGIBILITY_VERSION = "actor_answer_closed_teacher_supported_v2"


def empty_teacher_metadata() -> dict[str, Any]:
    return {
        "teacher_called": False,
        "teacher_total_call_count": 0,
        "teacher_answer": "",
        "teacher_evidence_status": "",
        "teacher_parse_status": "not_called",
        "teacher_format_error": False,
        "teacher_stage_b_called": False,
        "teacher_stage_b_used": False,
        "teacher_selected_stage": "",
        "teacher_selection_reason": "",
    }