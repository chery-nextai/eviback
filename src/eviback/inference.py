"""Actor-only iterative search inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from eviback.prompts import build_actor_messages
from eviback.retrieval.client import RetrieverClient

_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class ActorClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return one assistant turn."""


class OpenAICompatibleActorClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str = "",
        timeout_seconds: float = 180.0,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += (
                "/chat/completions" if self.endpoint.endswith("/v1") else "/v1/chat/completions"
            )
        self.model = model
        self.api_key = api_key or os.environ.get("EVIBACK_ACTOR_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"].get("content") or "")


def _action(text: str) -> tuple[str, str]:
    tools = list(_TOOL_RE.finditer(text))
    answers = list(_ANSWER_RE.finditer(text))
    if len(tools) + len(answers) != 1:
        return "error", "expected exactly one tool_call or answer"
    action_match = tools[0] if tools else answers[0]
    reasons = list(re.finditer(r"<(reason|think)>(.*?)</\1>", text, re.DOTALL))
    if len(reasons) != 1 or not reasons[0].group(2).strip():
        return "error", "expected exactly one non-empty reason block"
    reason = reasons[0]
    if (
        reason.start() >= action_match.start()
        or reason.end() > action_match.start()
        or text[: reason.start()].strip()
        or text[reason.end() : action_match.start()].strip()
        or text[action_match.end() :].strip()
    ):
        return "error", "reason/action order or surrounding text is invalid"
    if tools:
        try:
            payload = json.loads(tools[0].group(1))
            query = payload["arguments"]["query"]
            if payload.get("name") != "search" or not isinstance(query, str) or not query.strip():
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return "error", "invalid search tool call"
        return "search", query.strip()
    answer = answers[0].group(1).strip()
    return ("answer", answer) if answer else ("error", "empty answer")


def run_actor_inference(
    question: str,
    *,
    actor: ActorClient,
    retriever: RetrieverClient,
    retrieval_top_k: int = 50,
    visible_top_m: int = 5,
    max_turns: int = 10,
) -> dict[str, Any]:
    """Run inference without accepting a Teacher or reference-answer input."""

    messages = build_actor_messages(question)
    turns: list[dict[str, Any]] = []
    final_answer = ""
    status = "max_turns"
    for turn_index in range(1, max_turns + 1):
        assistant = actor.complete(messages)
        action, value = _action(assistant)
        messages.append({"role": "assistant", "content": assistant})
        if action == "answer":
            final_answer = value
            status = "answered"
            turns.append({"turn": turn_index, "assistant": assistant, "action": "answer"})
            break
        if action == "error":
            status = "malformed_actor_output"
            turns.append({"turn": turn_index, "assistant": assistant, "action": "error", "error": value})
            break
        documents = retriever.retrieve(value, top_k=retrieval_top_k)
        visible = [document.to_dict() for document in documents[:visible_top_m]]
        tool_response = json.dumps({"query": value, "documents": visible}, ensure_ascii=False)
        messages.append({"role": "user", "content": f"<tool_response>{tool_response}</tool_response>"})
        turns.append(
            {
                "turn": turn_index,
                "assistant": assistant,
                "action": "search",
                "query": value,
                "retrieved_evidence": visible,
            }
        )
    return {
        "schema_version": "eviback_actor_trajectory_v1",
        "question": question,
        "final_answer": final_answer,
        "status": status,
        "turn_count": len(turns),
        "search_count": sum(turn["action"] == "search" for turn in turns),
        "queries": [turn["query"] for turn in turns if turn["action"] == "search"],
        "turns": turns,
        "teacher_used": False,
        "reference_answer_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-endpoint", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--retriever-endpoint", required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument("--visible-top-m", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args(argv)
    actor = OpenAICompatibleActorClient(args.actor_endpoint, args.actor_model)
    retriever = RetrieverClient(args.retriever_endpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            result = run_actor_inference(
                str(row["question"]),
                actor=actor,
                retriever=retriever,
                retrieval_top_k=args.retrieval_top_k,
                visible_top_m=args.visible_top_m,
                max_turns=args.max_turns,
            )
            result.update(
                {
                    "index": row.get("index"),
                    "data_source": row.get("data_source", ""),
                }
            )
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())