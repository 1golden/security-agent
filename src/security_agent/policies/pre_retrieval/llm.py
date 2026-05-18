"""LLM-based pre-retrieval planner.

Sends a compact representation of the State to the LLM and asks for a
single Action in JSON. Falls back to NOOP if parsing fails.

Bug #4 fix: dynamically restrict the allowed action set based on what
the State already has — once the query has been rewritten, do NOT let
the LLM rewrite it again (that drift was the main cause of LLM-policy
losing to rule on the first baseline).
"""
from __future__ import annotations

import json

from security_agent.generation.llm_generator import LLMBackend
from security_agent.policies.base import PrePolicy
from security_agent.types import Action, ActionType, State


_ALL_ACTIONS = {"noop", "rewrite", "decompose", "inherit_filter", "choose_provider"}


def _allowed_actions(state: State) -> set[str]:
    """Bug #4: gate actions whose effect already applied."""
    allowed = set(_ALL_ACTIONS)
    if state.rewritten_query:
        allowed.discard("rewrite")
    if state.subqueries:
        allowed.discard("decompose")
    if state.provider_filter:
        allowed.discard("inherit_filter")
        allowed.discard("choose_provider")
    return allowed


_SYSTEM_TMPL = (
    "You are a query planner for a security-domain RAG agent. Given the "
    "user question, probe signals, and current state, choose ONE next action "
    "from: {allowed_csv}. Prefer 'noop' when the question is already clear. "
    "Reply with a JSON object: "
    '{{"action": "<name>", "payload": {{...}}, "rationale": "<one line>"}}. '
    "Keep payload minimal. Do NOT answer the question. "
    "Important: if 'rewrite' is not in the allowed list it is because the "
    "query was already rewritten this episode — do not try to rewrite again."
)


class LLMPrePolicy(PrePolicy):
    name = "pre.llm"

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def decide(self, state: State) -> Action:
        allowed = _allowed_actions(state)
        system = _SYSTEM_TMPL.format(allowed_csv=", ".join(sorted(allowed)))
        user = json.dumps(
            {
                "question": state.question.text,
                "probe": state.probe.to_dict(),
                "state": state.snapshot(),
                "attempt": state.attempt,
            },
            ensure_ascii=False,
        )
        data = self._llm.json(system, user)
        act = (data.get("action") or "noop").lower()
        if act not in allowed:
            # silently coerce a disallowed pick to noop rather than crash;
            # this keeps the trajectory valid for RL training later.
            return Action(
                type=ActionType.NOOP,
                rationale=f"llm picked disallowed action {act!r}; coerced to noop",
            )
        return Action(
            type=ActionType(act),
            payload=data.get("payload", {}) or {},
            rationale=data.get("rationale", "")[:200],
            confidence=float(data.get("confidence", 0.7)),
        )
