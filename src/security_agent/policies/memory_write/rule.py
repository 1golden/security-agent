"""Rule-based memory-write policy.

Decides — per axis — whether to write back. Risky axes (aliases, reflection)
default to OFF to limit the surface area for memory poisoning.
"""
from __future__ import annotations

from security_agent.policies.base import MemoryWritePolicy
from security_agent.types import (
    Action,
    ActionType,
    State,
    WRITE_AXIS_ALIASES,
    WRITE_AXIS_CONSTRAINTS,
    WRITE_AXIS_FACTS,
    WRITE_AXIS_PROVIDER_UTILITY,
    WRITE_AXIS_REFLECTION,
)


# How many turns must accumulate before we trigger an A-Mem-style reflection.
_REFLECTION_INTERVAL = 8


class RuleWritePolicy(MemoryWritePolicy):
    name = "write.rule"

    def decide(self, state: State) -> Action:
        # never write if we have no answer (failed run)
        if not state.answer or not state.finished:
            return Action(type=ActionType.SKIP_WRITE, rationale="no answer to anchor")

        axes: dict[str, bool] = {
            WRITE_AXIS_FACTS: state.coverage.score >= 0.4 and bool(state.documents),
            WRITE_AXIS_CONSTRAINTS: _has_constraint_signal(state),
            # aliases: only if coref was resolved in this turn AND coverage decent
            WRITE_AXIS_ALIASES: (
                bool(state.probe.coref_hints) and state.coverage.score >= 0.6
            ),
            # reflection: only every N turns, expensive
            WRITE_AXIS_REFLECTION: (
                state.attempt == 0
                and len(state.probe.recent_turns) >= _REFLECTION_INTERVAL
            ),
            # provider utility: write whenever retrieval used a specific provider
            WRITE_AXIS_PROVIDER_UTILITY: bool(state.provider_filter)
            and state.coverage.score >= 0.4,
        }

        if not any(axes.values()):
            return Action(type=ActionType.SKIP_WRITE, rationale="no axis crossed threshold")

        # build a concise content summary from the answer
        content = (state.answer or "")[:500]
        return Action(
            type=ActionType.WRITE,
            payload={
                "axes": {k: bool(v) for k, v in axes.items()},
                "content": content,
                "keywords": _extract_keywords(state),
                "providers": list(state.provider_filter),
                "evidence_ids": [d.id for d in state.documents[:5]],
            },
            rationale=f"writing axes: {[k for k, v in axes.items() if v]}",
        )


def _has_constraint_signal(state: State) -> bool:
    """Heuristic: phrases like 'do not', 'must', 'always' in the answer imply a rule."""
    if not state.answer:
        return False
    low = state.answer.lower()
    return any(
        marker in low
        for marker in ("must ", "never ", "always ", "do not", "shall ", "禁止", "必须")
    )


def _extract_keywords(state: State) -> list[str]:
    """Tiny extractor — first 5 capitalized tokens from the answer."""
    import re

    if not state.answer:
        return []
    caps = re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", state.answer)
    seen: list[str] = []
    for w in caps:
        if w not in seen:
            seen.append(w)
        if len(seen) >= 5:
            break
    return seen
