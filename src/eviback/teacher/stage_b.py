"""Conditional reference-calibration Stage B execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eviback.parsers import TeacherParseResult, parse_teacher_output
from eviback.prompts import STAGE_B_PROMPT_VERSION, build_stage_b_messages
from eviback.teacher.client import TeacherClient
from eviback.teacher.stage_a import TeacherStageResult, request_hash


def run_stage_b(
    client: TeacherClient,
    *,
    question: str,
    reference_answers: Sequence[Any],
    evidence_steps: Sequence[dict[str, Any]],
    top_m: int = 5,
) -> TeacherStageResult:
    messages = build_stage_b_messages(
        question, reference_answers, evidence_steps, top_m=top_m
    )
    digest = request_hash(messages, STAGE_B_PROMPT_VERSION)
    try:
        completion = client.complete(messages)
        return TeacherStageResult(
            stage="stage_b",
            prompt_version=STAGE_B_PROMPT_VERSION,
            called=True,
            parsed=parse_teacher_output(completion.content),
            raw_content=completion.content,
            messages=tuple(messages),
            request_hash=digest,
            elapsed_seconds=completion.elapsed_seconds,
        )
    except Exception as exc:
        return TeacherStageResult(
            stage="stage_b",
            prompt_version=STAGE_B_PROMPT_VERSION,
            called=True,
            parsed=TeacherParseResult(False, "", None, "", f"teacher_error:{type(exc).__name__}"),
            raw_content="",
            messages=tuple(messages),
            request_hash=digest,
            error=str(exc),
        )