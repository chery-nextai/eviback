"""Small OpenAI-compatible client boundary for the training-only Teacher."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TeacherCompletion:
    content: str
    elapsed_seconds: float = 0.0
    usage: Mapping[str, Any] = field(default_factory=dict)
    model: str = ""


class TeacherClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> TeacherCompletion:
        """Return one Teacher completion."""


class OpenAICompatibleTeacherClient:
    """Dependency-free client for `/v1/chat/completions` endpoints."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 180.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not str(endpoint).strip():
            raise ValueError("Teacher endpoint is required")
        if not str(model).strip():
            raise ValueError("Teacher model is required")
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += (
                "/chat/completions" if self.endpoint.endswith("/v1") else "/v1/chat/completions"
            )
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("EVIBACK_TEACHER_API_KEY", "")
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.extra_body = dict(extra_body or {})

    def complete(self, messages: Sequence[Mapping[str, str]]) -> TeacherCompletion:
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        try:
            message = result["choices"][0]["message"]
            content = str(message.get("content") or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Teacher response has no choices[0].message.content") from exc
        return TeacherCompletion(
            content=content,
            elapsed_seconds=time.perf_counter() - started,
            usage=dict(result.get("usage") or {}),
            model=str(result.get("model") or self.model),
        )


class MockTeacherClient:
    """Deterministic test client accepting an iterable or callback of outputs."""

    def __init__(
        self,
        responses: Iterable[str] | Callable[[Sequence[Mapping[str, str]], int], str],
    ) -> None:
        self._callback = responses if callable(responses) else None
        self._responses = iter(()) if callable(responses) else iter(responses)
        self.calls: list[list[dict[str, str]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(self, messages: Sequence[Mapping[str, str]]) -> TeacherCompletion:
        copied = [dict(message) for message in messages]
        self.calls.append(copied)
        if self._callback is not None:
            content = self._callback(copied, len(self.calls) - 1)
        else:
            try:
                content = next(self._responses)
            except StopIteration as exc:
                raise RuntimeError("MockTeacherClient has no remaining responses") from exc
        return TeacherCompletion(content=str(content), model="mock-teacher")