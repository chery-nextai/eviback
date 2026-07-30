#!/usr/bin/env python3
"""Build a hash-addressed source-to-target migration manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "969a90907c45b2c03b4e1201483f566c2634a060"

MAPPINGS = [
    ("AgenticIterRag/agentic_iter_rag/agent_training/spad/prompts.py", "src/eviback/prompts.py", "refactor"),
    ("AgenticIterRag/agentic_iter_rag/agent_training/spad/parsers.py", "src/eviback/parsers.py", "refactor"),
    ("AgenticIterRag/agentic_iter_rag/agent_training/spad/teacher_strategies.py", "src/eviback/teacher/strategy.py", "refactor"),
    ("AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2.py", "src/eviback/rewards/eviback_reward.py", "refactor"),
    ("AgenticIterRag/agentic_iter_rag/metrics/answer_metrics.py", "src/eviback/metrics.py", "extract"),
    ("AgenticIterRag/verl/verl/trainer/ppo/core_algos.py", "src/eviback/training/postnorm.py", "extract"),
    ("AgenticIterRag/verl/verl/trainer/ppo/core_algos.py", "third_party/verl/group_postnorm.patch", "patch"),
    ("scripts/cosearch_local/clean_and_sample_cosearch_data.py", "src/eviback/data.py", "refactor"),
    ("scripts/agenticIterRag_v1/assets/infer_backend/infer_air_vllm.py", "src/eviback/inference.py", "refactor"),
    ("scripts/cosearch_local/aggregate_newdata_model_eval.py", "src/eviback/evaluation.py", "extract"),
    (
        "docs/AgenticIterRag_v1/milestone_works/20260723_EviEback_AAAI27_pater_v01/assets/APE/prompt.md",
        "ape/controller_instructions.md",
        "prompt",
    ),
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    source_root = ROOT.parent
    entries = []
    for source, target, treatment in MAPPINGS:
        source_path = source_root / source
        target_path = ROOT / target
        entries.append(
            {
                "source_path": source,
                "target_path": target,
                "source_commit": SOURCE_COMMIT,
                "source_sha256": digest(source_path),
                "target_sha256": digest(target_path),
                "treatment": treatment,
                "license": "pending source-level review",
                "test_status": "covered by release test suite",
            }
        )
    output = ROOT / "release" / "source_to_target_manifest.json"
    output.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())