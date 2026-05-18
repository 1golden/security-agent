"""LLM backends + answer generator.

The same `LLMBackend` interface is used by policies (LLMPolicy) and by the
final answer generator. Two implementations:

- DummyLLM: deterministic stub, zero deps, used by tests and demo
- OpenAILLM: uses the `openai` package if installed
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from security_agent.config import LLMConfig
from security_agent.types import Document, MemoryRecord


@dataclass
class LLMResponse:
    text: str
    tokens: int = 0


class LLMBackend(ABC):
    """Common LLM interface. Subclasses track cumulative token usage on
    `total_tokens_used` so the pipeline can enforce a Budget that covers
    BOTH policy calls and the final answer generation (Bug #6 fix)."""

    def __init__(self) -> None:
        self.total_tokens_used: int = 0
        self.calls_made: int = 0

    @abstractmethod
    def _chat_impl(self, system: str, user: str, **kwargs) -> LLMResponse: ...

    def chat(self, system: str, user: str, **kwargs) -> LLMResponse:
        resp = self._chat_impl(system, user, **kwargs)
        self.total_tokens_used += int(resp.tokens or 0)
        self.calls_made += 1
        return resp

    def json(self, system: str, user: str, **kwargs) -> dict:
        """Convenience: ask for JSON, parse leniently."""
        resp = self.chat(system, user, **kwargs)
        text = resp.text.strip()
        # tolerate ```json fences
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        try:
            return json.loads(text)
        except Exception:
            return {}

    def reset_usage(self) -> None:
        self.total_tokens_used = 0
        self.calls_made = 0


class DummyLLM(LLMBackend):
    """Deterministic stub that echoes structure. Token count is char//4."""

    def _chat_impl(self, system: str, user: str, **kwargs) -> LLMResponse:
        head = user.strip().splitlines()[0][:120] if user.strip() else ""
        text = f"[dummy] system={system[:40]!r} user={head!r}"
        return LLMResponse(text=text, tokens=len(text) // 4)


class OpenAILLM(LLMBackend):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception as e:
            raise ImportError(
                "OpenAILLM requires `pip install openai`. Or use DummyLLM."
            ) from e
        kwargs: dict = {}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = OpenAI(**kwargs)
        self._model = cfg.model
        self._temp = cfg.temperature
        self._max_tokens = cfg.max_tokens

    def _chat_impl(self, system: str, user: str, **kwargs) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=kwargs.get("temperature", self._temp),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage else 0
        return LLMResponse(text=text, tokens=tokens)


def build_llm(cfg: LLMConfig) -> LLMBackend:
    backend = (cfg.backend or "dummy").lower()
    if backend == "dummy":
        return DummyLLM()
    if backend in ("openai", "azure"):
        return OpenAILLM(cfg)
    raise ValueError(f"Unknown LLM backend: {backend}")


class LLMGenerator:
    """Composes the final answer from documents + memory."""

    SYSTEM = (
        "You are a careful security-domain assistant. Answer the user using ONLY "
        "the provided evidence. Cite source ids in square brackets. If the evidence "
        "is insufficient, say so explicitly."
    )

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def generate(
        self,
        query: str,
        documents: list[Document],
        memory: list[MemoryRecord],
    ) -> tuple[str, int]:
        ev_lines: list[str] = []
        for d in documents:
            ev_lines.append(f"[{d.id}] ({d.source or '?'}) {d.text}")
        for m in memory:
            ev_lines.append(f"[mem:{m.id}] {m.content}")
        user = (
            f"Question: {query}\n\n"
            f"Evidence:\n" + ("\n".join(ev_lines) if ev_lines else "(none)") + "\n\n"
            "Answer:"
        )
        resp = self._llm.chat(self.SYSTEM, user)
        return resp.text, resp.tokens
