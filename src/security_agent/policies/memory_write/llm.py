"""LLM-based memory-write policy.

Asks the model to decide each axis independently. Gives the model an
explicit reminder that aliases/reflection are higher risk so it should be
conservative.
"""
from __future__ import annotations

import json

from security_agent.generation.llm_generator import LLMBackend
from security_agent.policies.base import MemoryWritePolicy
from security_agent.types import (
    Action,
    ActionType,
    State,
    ALL_WRITE_AXES,
    WRITE_AXIS_ALIASES,
    WRITE_AXIS_REFLECTION,
)


_SYSTEM = (
    "You are a memory-write gatekeeper. Decide, per axis, whether the agent "
    "should persist the latest turn into long-term memory. "
    f"Axes: {list(ALL_WRITE_AXES)}. "
    "Be CONSERVATIVE: skip the write entirely if the answer is uncertain. "
    f"Be EXTRA careful with {WRITE_AXIS_ALIASES!r} (poisoning risk) and "
    f"{WRITE_AXIS_REFLECTION!r} (expensive). "
    'Reply JSON: {"write": <bool>, "axes": {"facts": <bool>, ...}, '
    '"content": "<short summary or empty>", "rationale": "..."}'
)


class LLMWritePolicy(MemoryWritePolicy):
    name = "write.llm"

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def decide(self, state: State) -> Action:
        if not state.answer or not state.finished:
            return Action(type=ActionType.SKIP_WRITE, rationale="no answer")

        user = json.dumps(
            {
                "question": state.question.text,
                "answer": state.answer[:1500],
                "coverage": state.coverage.to_dict(),
                "n_documents": len(state.documents),
                "providers_used": list(state.provider_filter),
                "had_coref": bool(state.probe.coref_hints),
            },
            ensure_ascii=False,
        )
        data = self._llm.json(_SYSTEM, user)
        if not data.get("write"):
            return Action(
                type=ActionType.SKIP_WRITE,
                rationale=data.get("rationale", "")[:200] or "llm declined to write",
            )
        axes = {a: bool(data.get("axes", {}).get(a)) for a in ALL_WRITE_AXES}
        return Action(
            type=ActionType.WRITE,
            payload={
                "axes": axes,
                "content": (data.get("content") or state.answer)[:500],
                "providers": list(state.provider_filter),
                "evidence_ids": [d.id for d in state.documents[:5]],
            },
            rationale=data.get("rationale", "")[:200],
        )
