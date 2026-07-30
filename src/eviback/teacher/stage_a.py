"""Evidence-only Stage A execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from eviback.parsers import TeacherParseResult, parse_teacher_output
from eviback.prompts import STAGE_A_PROMPT_VERSION, build_stage_a_messages
from eviback.teacher.client import TeacherClient


@dataclass(frozen=True)
class TeacherStageResult:
    stage: str
    prompt_version: str
    called: bool
    parsed: TeacherParseResult
    raw_content: str
    messages: tuple[dict[str, str], ...]
    request_hash: str
    elapsed_seconds: float = 0.0
    error: str = ""

    def to_metadata(self, prefix: str) -> dict[str, Any]:
        return {
            f"teacher_{prefix}_called": self.called,
            f"teacher_{prefix}_prompt_version": self.prompt_version,
            f"teacher_{prefix}_answer": self.parsed.answer,
            f"teacher_{prefix}_evidence_status": self.parsed.status.value if self.parsed.status else "",
            f"teacher_{prefix}_parse_status": "parsed" if self.parsed.valid else self.parsed.error_code,
            f"teacher_{prefix}_format_error": not self.parsed.valid,
            f"teacher_{prefix}_raw_content": self.raw_content,
            f"teacher_{prefix}_messages": list(self.messages),
            f"teacher_{prefix}_request_hash": self.request_hash,
            f"teacher_{prefix}_elapsed_s": self.elapsed_seconds,
            f"teacher_{prefix}_error": self.error,
        }


def request_hash(messages: Sequence[dict[str, str]], prompt_version: str) -> str:
    identity = {"prompt_version": prompt_version, "messages": list(messages)}
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_stage_a(
    client: TeacherClient,
    *,
    question: str,
    evidence_steps: Sequence[dict[str, Any]],
    top_m: int = 5,
) -> TeacherStageResult:
    messages = build_stage_a_messages(question, evidence_steps, top_m=top_m)
    digest = request_hash(messages, STAGE_A_PROMPT_VERSION)
    try:
        completion = client.complete(messages)
        return TeacherStageResult(
            stage="stage_a",
            prompt_version=STAGE_A_PROMPT_VERSION,
            called=True,
            parsed=parse_teacher_output(completion.content),
            raw_content=completion.content,
            messages=tuple(messages),
            request_hash=digest,
            elapsed_seconds=completion.elapsed_seconds,
        )
    except Exception as exc:
        return TeacherStageResult(
            stage="stage_a",
            prompt_version=STAGE_A_PROMPT_VERSION,
            called=True,
            parsed=TeacherParseResult(False, "", None, "", f"teacher_error:{type(exc).__name__}"),
            raw_content="",
            messages=tuple(messages),
            request_hash=digest,
            error=str(exc),
        )