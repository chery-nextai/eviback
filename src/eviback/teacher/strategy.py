"""Public two-stage Evidence-Constrained Teacher strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from eviback.parsers import EvidenceStatus, TeacherParseResult
from eviback.prompts import STAGE_B_PROMPT_VERSION
from eviback.teacher.client import TeacherClient
from eviback.teacher.merge import MergeResult, merge_teacher_stages
from eviback.teacher.stage_a import TeacherStageResult, run_stage_a
from eviback.teacher.stage_b import run_stage_b

STRATEGY_ID = "spad_teacher_hard_gate_r5_literal_canonical_v2"


@dataclass(frozen=True)
class TeacherDecision:
    result: TeacherParseResult
    stage_a: TeacherStageResult
    stage_b: TeacherStageResult | None
    merge: MergeResult
    metadata: dict[str, Any]


class EvidenceConstrainedTeacher:
    """Run Stage A, conditionally calibrate in Stage B, and retain provenance."""

    def __init__(self, client: TeacherClient, *, visible_top_m: int = 5) -> None:
        if visible_top_m <= 0:
            raise ValueError("visible_top_m must be positive")
        self.client = client
        self.visible_top_m = int(visible_top_m)

    def judge(
        self,
        *,
        question: str,
        evidence_steps: Sequence[dict[str, Any]],
        reference_answers: Sequence[Any],
    ) -> TeacherDecision:
        stage_a = run_stage_a(
            self.client,
            question=question,
            evidence_steps=evidence_steps,
            top_m=self.visible_top_m,
        )
        parsed_non_insufficient = bool(
            stage_a.parsed.valid
            and stage_a.parsed.status
            in {EvidenceStatus.SUPPORTED, EvidenceStatus.AMBIGUOUS}
        )
        stage_b = (
            run_stage_b(
                self.client,
                question=question,
                reference_answers=reference_answers,
                evidence_steps=evidence_steps,
                top_m=self.visible_top_m,
            )
            if parsed_non_insufficient
            else None
        )
        merge = merge_teacher_stages(
            stage_a,
            stage_b,
            reference_answers=reference_answers,
            evidence_steps=evidence_steps,
        )
        metadata: dict[str, Any] = {
            "teacher_strategy_id": STRATEGY_ID,
            "teacher_total_call_count": 1 + int(stage_b is not None),
            "teacher_stage_b_called": stage_b is not None,
            "teacher_stage_b_used": merge.stage_b_used,
            "teacher_selected_stage": merge.selected_stage,
            "teacher_selection_reason": merge.selection_reason,
            "teacher_canonical_reference": merge.canonical_reference,
            "teacher_i_boundary_preserved": not (
                stage_a.parsed.status is EvidenceStatus.INSUFFICIENT
                and merge.selected.status is not EvidenceStatus.INSUFFICIENT
            ),
            **stage_a.to_metadata("stage_a"),
        }
        if stage_b is not None:
            metadata.update(stage_b.to_metadata("stage_b"))
        else:
            metadata.update(
                {
                    "teacher_stage_b_called": False,
                    "teacher_stage_b_prompt_version": STAGE_B_PROMPT_VERSION,
                    "teacher_stage_b_answer": "",
                    "teacher_stage_b_evidence_status": "",
                    "teacher_stage_b_parse_status": "not_called",
                    "teacher_stage_b_format_error": False,
                    "teacher_stage_b_raw_content": "",
                    "teacher_stage_b_messages": [],
                    "teacher_stage_b_request_hash": "",
                    "teacher_stage_b_elapsed_s": 0.0,
                    "teacher_stage_b_error": "",
                }
            )
        metadata.update(
            {
                "teacher_answer": merge.selected.answer,
                "teacher_evidence_status": (
                    merge.selected.status.value if merge.selected.status else ""
                ),
                "teacher_parse_status": (
                    "parsed" if merge.selected.valid else merge.selected.error_code
                ),
                "teacher_format_error": not merge.selected.valid,
            }
        )
        return TeacherDecision(merge.selected, stage_a, stage_b, merge, metadata)