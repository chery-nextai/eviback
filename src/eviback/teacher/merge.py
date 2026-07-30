"""Frozen merge rules for the two-stage Teacher."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from eviback.metrics import normalize_answer, token_f1
from eviback.parsers import EvidenceStatus, TeacherParseResult
from eviback.teacher.stage_a import TeacherStageResult


@dataclass(frozen=True)
class MergeResult:
    selected: TeacherParseResult
    selected_stage: str
    stage_b_used: bool
    selection_reason: str
    canonical_reference: str = ""


def find_literal_reference(
    evidence_steps: Sequence[dict[str, Any]], reference_answers: Sequence[Any]
) -> str:
    evidence_parts: list[str] = []
    for step in evidence_steps:
        for doc in step.get("docs") or []:
            evidence_parts.append(str(doc.get("title") or ""))
            evidence_parts.append(
                str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")
            )
    normalized_evidence = f" {normalize_answer(' '.join(evidence_parts))} "
    for reference in reference_answers:
        normalized_reference = normalize_answer(reference)
        if normalized_reference and f" {normalized_reference} " in normalized_evidence:
            return str(reference)
    return ""


def merge_teacher_stages(
    stage_a: TeacherStageResult,
    stage_b: TeacherStageResult | None,
    *,
    reference_answers: Sequence[Any],
    evidence_steps: Sequence[dict[str, Any]],
) -> MergeResult:
    """Merge without ever crossing a Stage A insufficiency boundary."""

    a = stage_a.parsed
    if not a.valid:
        return MergeResult(a, "stage_a", False, "stage_a_format_error")
    if a.status is EvidenceStatus.INSUFFICIENT:
        return MergeResult(a, "stage_a", False, "stage_a_insufficient_boundary")
    if stage_b is None or not stage_b.parsed.valid:
        return MergeResult(a, "stage_a", False, "stage_b_missing_or_invalid_fallback")
    b = stage_b.parsed
    if b.status is EvidenceStatus.INSUFFICIENT:
        return MergeResult(a, "stage_a", False, "stage_b_insufficient_fallback")
    if a.status is EvidenceStatus.SUPPORTED and b.status is not EvidenceStatus.SUPPORTED:
        return MergeResult(a, "stage_a", False, "stage_a_only_supported")

    selected = b
    selected_stage = "stage_b"
    reason = "stage_b_non_insufficient"
    if a.status is EvidenceStatus.SUPPORTED and b.status is EvidenceStatus.SUPPORTED:
        if token_f1(a.answer, reference_answers) > token_f1(b.answer, reference_answers):
            selected = a
            selected_stage = "stage_a"
            reason = "stage_a_supported_higher_reference_f1"

    canonical = ""
    if selected.status is EvidenceStatus.SUPPORTED:
        candidate = find_literal_reference(evidence_steps, reference_answers)
        if candidate and token_f1(candidate, reference_answers) > token_f1(
            selected.answer, reference_answers
        ):
            selected = TeacherParseResult(
                valid=True,
                reason=selected.reason,
                status=selected.status,
                answer=candidate,
            )
            canonical = candidate
            reason += "+evidence_literal_reference"
    return MergeResult(selected, selected_stage, selected_stage == "stage_b", reason, canonical)