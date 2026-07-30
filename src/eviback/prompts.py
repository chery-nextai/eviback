"""Frozen Actor and Evidence-Constrained Teacher prompts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

ACTOR_PROMPT_VERSION = "eviback_actor_search_r1_v1"
STAGE_A_PROMPT_VERSION = "spad_teacher_evidence_status_answer_v2"
STAGE_B_PROMPT_VERSION = "gold_support_evidence_only_v3"
INSUFFICIENT_ANSWER = "证据不足无法作答"

ACTOR_SYSTEM_PROMPT = """You are a tool-augmented research agent for factoid question answering.

Use the search tool before answering. Every assistant turn must contain a non-empty
<reason>...</reason> block followed by exactly one of:

<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>
<answer>short final answer</answer>

Do not emit a tool call and an answer in the same turn. Use a new query when more
evidence is needed. Inside <answer>, emit only the final answer span."""

_OUTPUT_CONTRACT = (
    "Output only three XML blocks in this exact order: "
    "<reason>...</reason><status>...</status><answer>...</answer>. "
    "The first character must be <. The status must be exactly one of "
    "supported_answer, insufficient_evidence, or ambiguous_evidence. "
    f"For insufficient_evidence or ambiguous_evidence, answer exactly {INSUFFICIENT_ANSWER}. "
    "For supported_answer, return the shortest answer span supported by the evidence."
)

STAGE_A_SYSTEM_PROMPT = (
    "You are an evidence-grounded QA model. Judge only the Original question from the "
    "Search evidence visible to the Actor. Do not use memorized facts. Use supported_answer "
    "only for one complete supported answer, insufficient_evidence when a necessary fact or "
    "bridge is absent, and ambiguous_evidence for multiple incompatible complete answers. "
    + _OUTPUT_CONTRACT
)

STAGE_B_SYSTEM_PROMPT = (
    "You are an evidence-grounded calibration judge. The Reference answer is only a candidate "
    "to verify and is never evidence. Verify the exact entity, predicate, scope, and every "
    "bridge against Search evidence. If the reference is unsupported, use a different complete "
    "evidence answer when one exists. "
    + _OUTPUT_CONTRACT
    + " Keep the reason under 60 words."
)


def _doc_text(doc: dict[str, Any]) -> str:
    return str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")


def _evidence_lines(
    evidence_steps: Sequence[dict[str, Any]], *, include_queries: bool, top_m: int
) -> list[str]:
    lines: list[str] = ["Search evidence:"]
    if not evidence_steps:
        return [*lines, "  (no search evidence provided)"]
    for round_index, step in enumerate(evidence_steps, start=1):
        lines.append(f"  Round {round_index}:")
        if include_queries:
            lines.append(f"    sub_query: {str(step.get('sub_query') or '')}")
        lines.append("    retrieved contents:")
        for doc_index, doc in enumerate((step.get("docs") or [])[:top_m], start=1):
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_index}"
            lines.append(f"      [{doc_index}] {title}")
            lines.append(f"        {_doc_text(doc)}")
    return lines


def build_actor_messages(question: str) -> list[dict[str, str]]:
    """Build the inference-time Actor prompt; it contains no Teacher or reference data."""

    return [
        {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": str(question).strip()},
    ]


def build_stage_a_messages(
    question: str, evidence_steps: Sequence[dict[str, Any]], *, top_m: int = 5
) -> list[dict[str, str]]:
    """Build Stage A messages from only question and Actor-visible evidence."""

    user = ["Original question:", str(question).strip(), ""]
    user.extend(_evidence_lines(evidence_steps, include_queries=True, top_m=top_m))
    user.extend(["", "Return the final XML result. Begin with <reason>."])
    return [
        {"role": "system", "content": STAGE_A_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user)},
    ]


def build_stage_b_messages(
    question: str,
    reference_answers: Sequence[Any],
    evidence_steps: Sequence[dict[str, Any]],
    *,
    top_m: int = 5,
) -> list[dict[str, str]]:
    """Build Stage B messages; reference answers are isolated from evidence."""

    references = [str(value) for value in reference_answers if str(value).strip()]
    user = [
        "Original question:",
        str(question).strip(),
        "",
        "Reference answer (candidate only, not evidence):",
        json.dumps(references, ensure_ascii=False),
        "",
    ]
    # Stage B intentionally hides retrieval sub-queries.
    user.extend(_evidence_lines(evidence_steps, include_queries=False, top_m=top_m))
    user.extend(["", "Return the final XML result. Begin with <reason>."])
    return [
        {"role": "system", "content": STAGE_B_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user)},
    ]