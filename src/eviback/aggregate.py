"""Paired comparison of already evaluated EviBack runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eviback.evaluation import METRIC_NAMES, paired_bootstrap


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any], *, samples: int = 10_000, seed: int = 42
) -> dict[str, Any]:
    if baseline["run_manifest"]["question_index_hash"] != candidate["run_manifest"]["question_index_hash"]:
        raise ValueError("run question-index hashes differ")
    baseline_rows = baseline["rows"]
    candidate_rows = candidate["rows"]
    baseline_keys = [(row["data_source"], str(row["index"])) for row in baseline_rows]
    candidate_keys = [(row["data_source"], str(row["index"])) for row in candidate_rows]
    if baseline_keys != candidate_keys:
        raise ValueError("run row order or identities differ")
    return {
        "schema_version": "eviback_paired_comparison_v1",
        "question_index_hash": baseline["run_manifest"]["question_index_hash"],
        "metrics": {
            metric: paired_bootstrap(
                [row[metric] for row in baseline_rows],
                [row[metric] for row in candidate_rows],
                samples=samples,
                seed=seed,
            )
            for metric in METRIC_NAMES
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    comparison = compare_reports(
        _load(args.baseline),
        _load(args.candidate),
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())