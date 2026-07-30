# Data

Raw QA data, the Wikipedia corpus, retrieval index, APE cases, and model weights
are not distributed here. Obtain NQ, HotpotQA, MuSiQue, 2WikiMultiHopQA,
TriviaQA, PopQA, and Bamboogle under their upstream terms, then arrange JSONL
files as `<raw-root>/<source>/{train,dev,test}.jsonl`.

Each record needs `question` plus `golden_answers`, `answers`, or `answer`.
Preparation removes duplicate normalized questions, drops empty answers, ranks
records by a stable SHA-256 key, and applies the frozen per-source quotas.

```bash
python scripts/prepare_data.py --raw-root /path/to/raw --output-root artifacts/data --format parquet
python scripts/prepare_data.py --check-only --manifest datasets-examples/manifests/train_5100.json
```

The checked-in manifests describe the paper artifacts without redistributing
them. A newly prepared artifact can differ in bytes when an upstream dataset
revision or Parquet library differs; record those revisions in the run report.