"""LLM-based pre-retrieval planner.

Sends a compact representation of the State to the LLM and asks for a
single Action in JSON. Falls back to NOOP if parsing fails.
"""
from __future__ import annotations

import json

from security_agent.generation.llm_generator import LLMBackend
from security_agent.policies.base import PrePolicy
from security_agent.types import Action, ActionType, State


_ALLOWED = {
    "noop",
    "rewrite",
    "decompose",
    "inherit_filter",
    "choose_provider",
}


_SYSTEM = (
    "You are a query planner for a security-domain RAG agent. Given the "
    "user question, probe signals, and current state, choose ONE next action "
    "from: noop, rewrite, decompose, inherit_filter, choose_provider. "
    "Reply with a JSON object: "
    '{"action": "<name>", "payload": {...}, "rationale": "<one line>"}. '
    "Keep payload minimal. Do NOT answer the question."
)


class LLMPrePolicy(PrePolicy):
    name = "pre.llm"

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def decide(self, state: State) -> Action:
        user = json.dumps(
            {
                "question": state.question.text,
                "probe": state.probe.to_dict(),
                "state": state.snapshot(),
            },
            ensure_ascii=False,
        )
        data = self._llm.json(_SYSTEM, user)
        act = (data.get("action") or "noop").lower()
        if act not in _ALLOWED:
            return Action(type=ActionType.NOOP, rationale="llm returned invalid action")
        return Action(
            type=ActionType(act),
            payload=data.get("payload", {}) or {},
            rationale=data.get("rationale", "")[:200],
            confidence=float(data.get("confidence", 0.7)),
        )
