"""Deterministic, license-conscious data preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

TRAIN_QUOTAS = {
    "nq": 2040,
    "hotpotqa": 1416,
    "musique": 897,
    "2wikimultihopqa": 747,
}
EVAL_QUOTAS = {
    "popqa": 563,
    "2wikimultihopqa": 563,
    "triviaqa": 563,
    "hotpotqa": 562,
    "nq": 562,
    "musique": 562,
    "bamboogle": 125,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(value.split())


def normalize_answers(item: dict[str, Any]) -> list[str]:
    raw = item.get("golden_answers")
    if raw is None:
        raw = item.get("answers")
    if raw is None:
        raw = [item["answer"]] if item.get("answer") is not None else []
    if isinstance(raw, str):
        raw = [raw]
    answers: list[str] = []
    seen: set[str] = set()
    for answer in raw or []:
        text = str(answer).strip()
        key = normalize_text(text)
        if text and key and key not in seen:
            answers.append(text)
            seen.add(key)
    return answers


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_key(item: dict[str, Any], source: str, seed: int) -> str:
    identity = canonical_json(
        {
            "seed": seed,
            "source": source,
            "id": item.get("id") or item.get("source_id") or "",
            "question": normalize_text(item.get("question")),
        }
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def stable_sample(
    rows: Sequence[dict[str, Any]], source: str, count: int, seed: int
) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_key = normalize_text(row.get("question"))
        if question_key and normalize_answers(row):
            current = by_question.get(question_key)
            if current is None or stable_key(row, source, seed) < stable_key(current, source, seed):
                by_question[question_key] = row
    ordered = sorted(by_question.values(), key=lambda item: stable_key(item, source, seed))
    if len(ordered) < count:
        raise ValueError(f"{source}: requested {count} rows but only {len(ordered)} valid unique rows")
    return ordered[:count]


def public_row(item: dict[str, Any], source: str, split: str, index: int) -> dict[str, Any]:
    question = " ".join(str(item["question"]).strip().split())
    return {
        "data_source": source,
        "question": question,
        "answers": normalize_answers(item),
        "split": split,
        "index": index,
        "source_id": str(item.get("id") or item.get("source_id") or index),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_dataset(path: Path, rows: list[dict[str, Any]], output_format: str) -> Path:
    if output_format == "jsonl":
        write_jsonl(path.with_suffix(".jsonl"), rows)
        return path.with_suffix(".jsonl")
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet output requires the eviback[data] dependencies") from exc
    output = path.with_suffix(".parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output


def prepare_split(
    raw_root: Path,
    output_root: Path,
    *,
    split: str,
    quotas: dict[str, int],
    seed: int,
    output_format: str,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for source, quota in quotas.items():
        filename = "train.jsonl" if split == "train" else (
            "test.jsonl" if source in {"nq", "triviaqa", "popqa", "bamboogle"} else "dev.jsonl"
        )
        input_path = raw_root / source / filename
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        sampled = stable_sample(read_jsonl(input_path), source, quota, seed)
        output_rows.extend(
            public_row(item, source, split, index) for index, item in enumerate(sampled)
        )
        source_counts[source] = len(sampled)
    output_rows.sort(key=lambda row: stable_key(row, row["data_source"], seed))
    base = output_root / ("train_5100" if split == "train" else "eval_3500")
    output_path = write_dataset(base, output_rows, output_format)
    manifest = {
        "schema_version": "eviback_data_v1",
        "kind": split,
        "size": len(output_rows),
        "seed": seed,
        "sampling": "stable SHA-256 rank per source after question deduplication",
        "source_quotas": dict(sorted(source_counts.items())),
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
    }
    manifest_path = output_root / f"{split}_{len(output_rows)}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"kind", "size", "seed", "source_quotas"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"manifest is missing fields: {missing}")
    if sum(int(value) for value in manifest["source_quotas"].values()) != int(manifest["size"]):
        raise ValueError("manifest source quotas do not sum to size")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--split", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--format", choices=("jsonl", "parquet"), default="jsonl")
    parser.add_argument("--train-seed", type=int, default=26041755)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.check_only:
        if args.manifest is None:
            parser.error("--check-only requires --manifest")
        print(json.dumps(validate_manifest(args.manifest), indent=2, sort_keys=True))
        return 0
    if args.raw_root is None:
        parser.error("--raw-root is required unless --check-only is used")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.split in {"train", "both"}:
        prepare_split(
            args.raw_root,
            args.output_root,
            split="train",
            quotas=TRAIN_QUOTAS,
            seed=args.train_seed,
            output_format=args.format,
        )
    if args.split in {"eval", "both"}:
        prepare_split(
            args.raw_root,
            args.output_root,
            split="eval",
            quotas=EVAL_QUOTAS,
            seed=args.eval_seed,
            output_format=args.format,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())