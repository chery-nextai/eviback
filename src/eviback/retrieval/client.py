"""HTTP client for the Search-R1-compatible local retriever."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    title: str
    contents: str
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrieverClient:
    def __init__(self, endpoint: str, *, timeout_seconds: float = 60.0) -> None:
        if not str(endpoint).strip():
            raise ValueError("retriever endpoint is required")
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/retrieve"):
            self.endpoint += "/retrieve"
        self.timeout_seconds = float(timeout_seconds)

    def retrieve(self, query: str, *, top_k: int = 50) -> list[RetrievedDocument]:
        if not str(query).strip():
            raise ValueError("query must be non-empty")
        payload = {"queries": [query], "topk": int(top_k), "return_scores": True}
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        batches = result.get("result")
        if not isinstance(batches, list) or not batches or not isinstance(batches[0], list):
            raise ValueError("retriever response must contain result[0] list")
        documents: list[RetrievedDocument] = []
        for rank, raw in enumerate(batches[0], start=1):
            item = raw.get("document", raw) if isinstance(raw, dict) else {}
            score = raw.get("score", item.get("score", 0.0)) if isinstance(raw, dict) else 0.0
            documents.append(
                RetrievedDocument(
                    doc_id=str(item.get("id") or item.get("doc_id") or ""),
                    title=str(item.get("title") or ""),
                    contents=str(item.get("contents") or item.get("text") or item.get("passage") or ""),
                    score=float(score or 0.0),
                    rank=rank,
                )
            )
        return documents