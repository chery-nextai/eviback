"""Aligned evaluation, macro metrics, and paired bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eviback.metrics import exact_match, token_f1

SINGLE_HOP_SOURCES = frozenset({"nq", "triviaqa", "popqa"})
MULTI_HOP_SOURCES = frozenset({"hotpotqa", "2wikimultihopqa", "musique", "bamboogle"})
METRIC_NAMES = (
    "legacy_em",
    "token_f1",
    "valid_answer_rate",
    "mean_search_calls",
    "duplicate_query_rate",
    "maximum_turn_rate",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("data_source") or ""), str(row.get("index"))


def align_predictions(
    predictions: Sequence[dict[str, Any]], references: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    prediction_map: dict[tuple[str, str], dict[str, Any]] = {}
    reference_map: dict[tuple[str, str], dict[str, Any]] = {}
    for name, rows, destination in (
        ("prediction", predictions, prediction_map),
        ("reference", references, reference_map),
    ):
        for row in rows:
            key = _key(row)
            if key in destination:
                raise ValueError(f"duplicate {name} key: {key}")
            destination[key] = row
    if set(prediction_map) != set(reference_map):
        missing = sorted(set(reference_map) - set(prediction_map))[:5]
        extra = sorted(set(prediction_map) - set(reference_map))[:5]
        raise ValueError(f"prediction/reference keys differ; missing={missing}, extra={extra}")
    aligned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key in sorted(reference_map):
        prediction = prediction_map[key]
        reference = reference_map[key]
        if str(prediction.get("question") or "").strip() != str(reference.get("question") or "").strip():
            raise ValueError(f"question text differs for key {key}")
        aligned.append((prediction, reference))
    return aligned


def _answers(reference: dict[str, Any]) -> list[str]:
    raw = reference.get("answers") or reference.get("reference_answers") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(value) for value in raw]


def score_aligned(
    aligned: Sequence[tuple[dict[str, Any], dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction, reference in aligned:
        answer = str(prediction.get("final_answer") or prediction.get("answer") or "").strip()
        queries = [" ".join(str(value).casefold().split()) for value in prediction.get("queries") or []]
        status = str(prediction.get("status") or "")
        rows.append(
            {
                "data_source": str(reference.get("data_source") or ""),
                "index": reference.get("index"),
                "question": reference.get("question"),
                "legacy_em": exact_match(answer, _answers(reference)),
                "token_f1": token_f1(answer, _answers(reference)),
                "valid_answer_rate": float(bool(answer) and status == "answered"),
                "mean_search_calls": float(prediction.get("search_count", len(queries)) or 0),
                "duplicate_query_rate": float(len(queries) > len(set(queries))),
                "maximum_turn_rate": float(status == "max_turns"),
            }
        )
    return rows


def _mean(rows: Sequence[dict[str, Any]], metric: str) -> float:
    return sum(float(row[metric]) for row in rows) / len(rows) if rows else 0.0


def _macro(rows: Sequence[dict[str, Any]], sources: frozenset[str]) -> dict[str, float]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["data_source"] in sources:
            by_source[row["data_source"]].append(row)
    return {
        metric: sum(_mean(source_rows, metric) for source_rows in by_source.values()) / len(by_source)
        if by_source
        else 0.0
        for metric in METRIC_NAMES
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["data_source"])].append(row)
    return {
        "count": len(rows),
        "overall": {metric: _mean(rows, metric) for metric in METRIC_NAMES},
        "by_source": {
            source: {"count": len(source_rows), **{metric: _mean(source_rows, metric) for metric in METRIC_NAMES}}
            for source, source_rows in sorted(by_source.items())
        },
        "single_hop_macro": _macro(rows, SINGLE_HOP_SOURCES),
        "multi_hop_macro": _macro(rows, MULTI_HOP_SOURCES),
    }


def question_index_hash(rows: Sequence[dict[str, Any]]) -> str:
    identities = sorted(
        [
        {
            "data_source": row.get("data_source"),
            "index": row.get("index"),
            "question": row.get("question"),
        }
        for row in rows
        ],
        key=lambda row: (str(row["data_source"]), str(row["index"]), str(row["question"])),
    )
    encoded = json.dumps(identities, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap requires equally sized non-empty vectors")
    rng = np.random.default_rng(seed)
    delta = right - left
    # Generate in bounded chunks so the formal 3,500 x 10,000 run stays memory-light.
    means: list[np.ndarray] = []
    remaining = int(samples)
    while remaining:
        count = min(remaining, 1000)
        draws = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[draws].mean(axis=1))
        remaining -= count
    bootstrap_means = np.concatenate(means)
    return {
        "candidate_minus_baseline": float(delta.mean()),
        "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-checkpoint", default="")
    parser.add_argument("--retriever-revision", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    references = read_jsonl(args.references)
    scored = score_aligned(align_predictions(read_jsonl(args.predictions), references))
    report = {
        "schema_version": "eviback_evaluation_v1",
        "run_manifest": {
            "question_index_hash": question_index_hash(references),
            "model_checkpoint": args.model_checkpoint,
            "retriever_revision": args.retriever_revision,
            "seed": args.seed,
            "teacher_used": False,
            "reference_used_during_inference": False,
        },
        "summary": summarize(scored),
        "rows": scored,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())