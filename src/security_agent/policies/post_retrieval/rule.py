"""Rule-based post-retrieval verifier.

Reads Coverage + Budget to decide one of:
  answer | filter | gap_retry | broaden | narrow | stop
"""
from __future__ import annotations

from security_agent.policies.base import PostPolicy
from security_agent.types import Action, ActionType, State


class RulePostPolicy(PostPolicy):
    name = "post.rule"

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def decide(self, state: State) -> Action:
        cov = state.coverage
        budget = state.budget
        delta = state.coverage_delta()

        # 1) coverage good enough → answer
        if cov.score >= self._threshold and not cov.missing_aspects:
            return Action(
                type=ActionType.ANSWER,
                rationale=f"coverage {cov.score:.2f} >= {self._threshold}",
            )

        # 2) too redundant → narrow (try a tighter query)
        if cov.redundancy > 0.7 and budget.can_retry(state.attempt, delta):
            return Action(
                type=ActionType.NARROW,
                payload={"reason": "high redundancy"},
                rationale=f"redundancy {cov.redundancy:.2f}",
            )

        # 3) specific gap → targeted gap_retry on the first missing aspect
        if cov.missing_aspects and budget.can_retry(state.attempt, delta):
            return Action(
                type=ActionType.GAP_RETRY,
                payload={"aspect": cov.missing_aspects[0]},
                rationale=f"targeting missing aspect: {cov.missing_aspects[0]}",
            )

        # 4) low score, no specific gap → broaden
        if cov.score < self._threshold and budget.can_retry(state.attempt, delta):
            return Action(
                type=ActionType.BROADEN,
                payload={"reason": "low overall coverage"},
                rationale=f"coverage {cov.score:.2f} below threshold",
            )

        # 5) low confidence claims and budget left → filter them out
        if cov.low_confidence_claims and budget.can_retry(state.attempt, delta):
            return Action(
                type=ActionType.FILTER,
                payload={"drop": list(cov.low_confidence_claims)},
                rationale="dropping low-confidence claims",
            )

        # 6) budget exhausted / no improvement → stop (answer with what we have)
        if state.documents:
            return Action(
                type=ActionType.ANSWER,
                rationale="budget exhausted, answering with available evidence",
            )
        return Action(type=ActionType.STOP, rationale="no evidence, budget exhausted")
