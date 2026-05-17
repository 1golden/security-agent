"""Rule-based pre-retrieval planner.

Heuristics, no LLM. Good baseline + deterministic for tests.
"""
from __future__ import annotations

import re

from security_agent.policies.base import PrePolicy
from security_agent.types import Action, ActionType, State


# Markers that suggest the user wants two distinct things in one query.
DECOMPOSE_MARKERS = re.compile(r"\b(and|then|also|as well as|;|，|；)\b", re.IGNORECASE)


class RulePrePolicy(PrePolicy):
    name = "pre.rule"

    def decide(self, state: State) -> Action:
        q = state.question.text.strip()
        probe = state.probe

        # 1) inherit a filter the probe already detected, no rewrite cost
        if probe.explicit_filters and not state.inherited_filters:
            return Action(
                type=ActionType.INHERIT_FILTER,
                payload={"filters": dict(probe.explicit_filters)},
                rationale="explicit filter detected by probe",
            )

        # 2) if probe surfaced a preferred provider but no filter set yet, bias
        if probe.preferred_providers and not state.provider_filter:
            return Action(
                type=ActionType.CHOOSE_PROVIDER,
                payload={"providers": probe.preferred_providers[:2]},
                rationale="reusing providers that helped in past sessions",
            )

        # 3) coref present and not yet resolved → rewrite to expand the pronoun
        if probe.coref_hints and not state.rewritten_query:
            rewritten = q
            for pronoun, entity in probe.coref_hints.items():
                rewritten = re.sub(rf"\b{pronoun}\b", entity, rewritten, flags=re.IGNORECASE)
            if rewritten != q:
                return Action(
                    type=ActionType.REWRITE,
                    payload={"query": rewritten},
                    rationale=f"resolved coref {probe.coref_hints!r}",
                )

        # 4) compound question → decompose
        if not state.subqueries and DECOMPOSE_MARKERS.search(q) and len(q) > 30:
            parts = re.split(DECOMPOSE_MARKERS, q)
            parts = [p.strip(" ,;，；") for p in parts if p and p.strip()]
            parts = [p for p in parts if len(p) > 5]
            if len(parts) >= 2:
                return Action(
                    type=ActionType.DECOMPOSE,
                    payload={"subqueries": parts[:4]},
                    rationale="conjunctive markers detected",
                )

        # 5) nothing to do
        return Action(type=ActionType.NOOP, rationale="planning idle")
