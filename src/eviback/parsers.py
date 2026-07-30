"""Conservative parsers for Teacher XML and Search-R1 trajectories."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from eviback.prompts import INSUFFICIENT_ANSWER


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported_answer"
    INSUFFICIENT = "insufficient_evidence"
    AMBIGUOUS = "ambiguous_evidence"


@dataclass(frozen=True)
class TeacherParseResult:
    valid: bool
    reason: str
    status: EvidenceStatus | None
    answer: str
    error_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value if self.status else ""
        return result


@dataclass(frozen=True)
class ActorTrajectoryParseResult:
    valid: bool
    answer: str
    queries: tuple[str, ...]
    assistant_turn_count: int
    status: str
    error_code: str = ""

    @property
    def search_count(self) -> int:
        return len(self.queries)

    @property
    def duplicate_query_count(self) -> int:
        normalized = [normalize_query(query) for query in self.queries if normalize_query(query)]
        return len(normalized) - len(set(normalized))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["search_count"] = self.search_count
        result["duplicate_query_count"] = self.duplicate_query_count
        return result


_TEACHER_RE = re.compile(
    r"<reason>(.*?)</reason>\s*<status>(.*?)</status>\s*<answer>(.*?)</answer>",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def normalize_query(query: Any) -> str:
    return " ".join(str(query or "").casefold().split())


def parse_teacher_output(text: Any) -> TeacherParseResult:
    """Parse one strict XML result; truncated or extra output is invalid."""

    raw = str(text or "").strip()
    if not raw:
        return TeacherParseResult(False, "", None, "", "empty_output")
    for tag in ("reason", "status", "answer"):
        open_count = raw.count(f"<{tag}>")
        close_count = raw.count(f"</{tag}>")
        if open_count == 0 or close_count == 0:
            return TeacherParseResult(False, "", None, "", f"missing_{tag}_tag")
        if open_count != 1 or close_count != 1:
            return TeacherParseResult(False, "", None, "", f"invalid_{tag}_tag_count")
    matches = list(_TEACHER_RE.finditer(raw))
    if len(matches) != 1:
        if "<reason>" not in raw or "</reason>" not in raw:
            code = "missing_or_truncated_reason"
        elif "<status>" not in raw or "</status>" not in raw:
            code = "missing_or_truncated_status"
        elif "<answer>" not in raw or "</answer>" not in raw:
            code = "missing_or_truncated_answer"
        else:
            code = "multiple_or_misordered_blocks"
        return TeacherParseResult(False, "", None, "", code)
    match = matches[0]
    if raw[: match.start()].strip() or raw[match.end() :].strip():
        return TeacherParseResult(False, "", None, "", "extra_text_outside_xml")
    reason, raw_status, answer = (part.strip() for part in match.groups())
    if not reason:
        return TeacherParseResult(False, reason, None, answer, "empty_reason")
    try:
        status = EvidenceStatus(raw_status)
    except ValueError:
        return TeacherParseResult(False, reason, None, answer, "invalid_status")
    if not answer or answer in {"...", "…"}:
        return TeacherParseResult(False, reason, status, answer, "empty_or_placeholder_answer")
    if status in {EvidenceStatus.INSUFFICIENT, EvidenceStatus.AMBIGUOUS} and answer != INSUFFICIENT_ANSWER:
        return TeacherParseResult(False, reason, status, answer, "status_answer_contract_error")
    if status is EvidenceStatus.SUPPORTED and answer == INSUFFICIENT_ANSWER:
        return TeacherParseResult(False, reason, status, answer, "status_answer_contract_error")
    return TeacherParseResult(True, reason, status, answer)


def _assistant_blocks(text: str) -> list[str]:
    chatml = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if chatml:
        return [block.strip() for block in chatml if block.strip()]
    # Internal traces delimit tool responses with a user turn and later assistant marker.
    blocks = re.split(r"\nuser\n<tool_response>.*?</tool_response>\nassistant\n", text, flags=re.DOTALL)
    return [block.removeprefix("assistant\n").strip() for block in blocks if block.strip()]


def _reason_is_valid(block: str, action_start: int, action_end: int) -> bool:
    matches = list(re.finditer(r"<(reason|think)>(.*?)</\1>", block, re.DOTALL))
    if len(matches) != 1:
        return False
    match = matches[0]
    return bool(
        match.start() < action_start
        and match.end() <= action_start
        and match.group(2).strip()
        and not block[: match.start()].strip()
        and not block[match.end() : action_start].strip()
        and not block[action_end:].strip()
    )


def parse_actor_trajectory(text: Any, *, max_turns: int | None = None) -> ActorTrajectoryParseResult:
    """Validate tool calls and the final closed answer conservatively."""

    blocks = _assistant_blocks(str(text or "").strip())
    if not blocks:
        return ActorTrajectoryParseResult(False, "", (), 0, "malformed", "empty_trajectory")
    queries: list[str] = []
    for turn_index, block in enumerate(blocks):
        tool_matches = list(_TOOL_RE.finditer(block))
        answer_matches = list(_ANSWER_RE.finditer(block))
        is_final = turn_index == len(blocks) - 1
        if tool_matches and answer_matches:
            return ActorTrajectoryParseResult(
                False, "", tuple(queries), len(blocks), "malformed", "tool_and_answer_same_turn"
            )
        if tool_matches:
            if len(tool_matches) != 1 or not _reason_is_valid(
                block, tool_matches[0].start(), tool_matches[0].end()
            ):
                return ActorTrajectoryParseResult(
                    False, "", tuple(queries), len(blocks), "malformed", "invalid_tool_turn"
                )
            try:
                payload = json.loads(tool_matches[0].group(1))
                arguments = payload.get("arguments")
                query = arguments.get("query") if isinstance(arguments, dict) else None
                if payload.get("name") != "search" or not isinstance(query, str) or not query.strip():
                    raise ValueError
            except (ValueError, TypeError, json.JSONDecodeError):
                return ActorTrajectoryParseResult(
                    False, "", tuple(queries), len(blocks), "malformed", "invalid_tool_call"
                )
            queries.append(query.strip())
            if is_final:
                return ActorTrajectoryParseResult(
                    False, "", tuple(queries), len(blocks), "no_finish", "final_turn_is_tool_call"
                )
            continue
        if answer_matches:
            if not is_final or len(answer_matches) != 1:
                return ActorTrajectoryParseResult(
                    False, "", tuple(queries), len(blocks), "malformed", "answer_not_unique_final"
                )
            if not _reason_is_valid(
                block, answer_matches[0].start(), answer_matches[0].end()
            ):
                return ActorTrajectoryParseResult(
                    False, "", tuple(queries), len(blocks), "malformed", "invalid_answer_turn"
                )
            answer = answer_matches[0].group(1).strip()
            if not answer:
                return ActorTrajectoryParseResult(
                    False, "", tuple(queries), len(blocks), "malformed", "empty_answer"
                )
            status = "max_turns" if max_turns is not None and len(blocks) >= max_turns else "answered"
            return ActorTrajectoryParseResult(True, answer, tuple(queries), len(blocks), status)
        return ActorTrajectoryParseResult(
            False, "", tuple(queries), len(blocks), "malformed", "missing_action"
        )
    return ActorTrajectoryParseResult(False, "", tuple(queries), len(blocks), "no_finish", "missing_answer")


def extract_last_answer(text: Any) -> str:
    matches = list(_ANSWER_RE.finditer(str(text or "")))
    return matches[-1].group(1).strip() if matches else ""