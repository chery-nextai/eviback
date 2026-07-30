"""Optional E5 + FAISS retriever server implementing the public contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def create_app(index_path: Path, corpus_path: Path, model_name: str, device: str, default_top_k: int):
    try:
        import faiss
        import numpy as np
        import torch
        from fastapi import FastAPI
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install eviback[retrieval] to start the retriever") from exc

    documents = load_documents(corpus_path)
    index = faiss.read_index(str(index_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=False).to(device).eval()

    @torch.no_grad()
    def encode(queries: list[str]):
        inputs = tokenizer(
            [f"query: {query}" for query in queries],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hidden = model(**inputs, return_dict=True).last_hidden_state
        mask = inputs["attention_mask"]
        pooled = hidden.masked_fill(~mask[..., None].bool(), 0).sum(1) / mask.sum(1)[..., None]
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.float().cpu().numpy().astype(np.float32)

    app = FastAPI(title="EviBack retriever")

    @app.post("/retrieve")
    def retrieve(request: dict[str, Any]):
        queries = request.get("queries")
        if not isinstance(queries, list) or not queries or not all(
            isinstance(query, str) and query.strip() for query in queries
        ):
            raise ValueError("queries must be a non-empty list of strings")
        top_k = int(request.get("topk") or default_top_k)
        scores, indices = index.search(encode(queries), top_k)
        result = []
        for row_scores, row_indices in zip(scores, indices):
            batch = []
            for score, index_value in zip(row_scores, row_indices):
                document = dict(documents[int(index_value)])
                if request.get("return_scores", False):
                    batch.append({"document": document, "score": float(score)})
                else:
                    document["score"] = float(score)
                    batch.append(document)
            result.append(batch)
        return {"result": result}

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install eviback[retrieval] to start the retriever") from exc
    uvicorn.run(
        create_app(args.index, args.corpus, args.model, args.device, args.top_k),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())