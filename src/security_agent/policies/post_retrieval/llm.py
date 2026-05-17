"""LLM-based post-retrieval verifier."""
from __future__ import annotations

import json

from security_agent.generation.llm_generator import LLMBackend
from security_agent.policies.base import PostPolicy
from security_agent.types import Action, ActionType, State


_ALLOWED = {"answer", "filter", "gap_retry", "broaden", "narrow", "stop"}

_SYSTEM = (
    "You are a retrieval-quality judge. Given the question, retrieved docs, "
    "structured coverage, and remaining budget, pick ONE action: "
    "answer / filter / gap_retry / broaden / narrow / stop. "
    "Prefer answer when coverage is acceptable. Use gap_retry when a specific "
    "aspect is missing. Use broaden when score is low and no specific gap is "
    "named. Use narrow when redundancy is high. Use stop only when budget is "
    "exhausted and no further retry can help. "
    'Reply JSON: {"action": "<name>", "payload": {...}, "rationale": "..."}'
)


class LLMPostPolicy(PostPolicy):
    name = "post.llm"

    def __init__(self, llm: LLMBackend, threshold: float = 0.5) -> None:
        self._llm = llm
        self._threshold = threshold

    def decide(self, state: State) -> Action:
        user = json.dumps(
            {
                "question": state.question.text,
                "coverage": state.coverage.to_dict(),
                "n_documents": len(state.documents),
                "doc_sources": [d.source for d in state.documents],
                "budget": state.budget.to_dict(),
                "attempt": state.attempt,
                "threshold": self._threshold,
            },
            ensure_ascii=False,
        )
        data = self._llm.json(_SYSTEM, user)
        act = (data.get("action") or "answer").lower()
        if act not in _ALLOWED:
            act = "answer"
        return Action(
            type=ActionType(act),
            payload=data.get("payload", {}) or {},
            rationale=data.get("rationale", "")[:200],
            confidence=float(data.get("confidence", 0.7)),
        )
