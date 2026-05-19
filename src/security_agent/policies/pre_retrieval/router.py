"""Router PrePolicy — picks Rule or LLM per question by cheap features.

Motivated by the v2 baseline finding that LLM-policy wins on some
questions (Q1 OWASP A03: +32, Q8 CISA KEV: +56) and loses on others
(Q2/Q6/Q7: -65 to -90), with the gap dominated by variance, not the
mean. An ensemble that picks the right policy per question can beat
either pure policy.

This implementation uses a 5-feature heuristic (no training, no LLM
call at decision time) that maps question shape → policy choice. The
features are:

  1. has_compound: question contains a conjunction ("and"/"then"/...)
     → LLM tends to do useful decompose
  2. has_pronoun: question has a context pronoun ("it"/"they"/...)
     → LLM coref handling can help
  3. is_short_factoid: ≤6 word lookup-style question
     → Rule's "do nothing" baseline wins
  4. is_definition: starts with "What is/are/does ..."
     → Rule wins (don't over-rewrite)
  5. is_open_howto: starts with "How"
     → LLM's gap-retry is more useful

The same router can later be swapped for a learned classifier trained
on (question, which_policy_won) pairs from the 50-Q benchmark. The
interface is identical so the swap is transparent to the pipeline.
"""
from __future__ import annotations

import re

from security_agent.policies.base import PrePolicy
from security_agent.policies.pre_retrieval.llm import LLMPrePolicy
from security_agent.policies.pre_retrieval.rule import RulePrePolicy
from security_agent.types import Action, State


_PRONOUNS = ("it", "they", "this", "that", "these", "those", "he", "she")
_CONJ_RE = re.compile(r"\b(and|then|also|as well as)\b", re.IGNORECASE)
_HOW_RE = re.compile(r"^\s*how\b", re.IGNORECASE)
_DEF_RE = re.compile(r"^\s*what\s+(is|are|does|do)\b", re.IGNORECASE)


def _features(q: str) -> dict[str, bool]:
    low = q.lower()
    word_count = len(re.findall(r"\S+", q))
    return {
        "has_compound": bool(_CONJ_RE.search(q)) and word_count >= 6,
        "has_pronoun": any(re.search(rf"\b{p}\b", low) for p in _PRONOUNS),
        "is_short_factoid": word_count <= 6,
        "is_definition": bool(_DEF_RE.match(q)),
        "is_open_howto": bool(_HOW_RE.match(q)),
    }


def choose(q: str) -> str:
    """Return the name of the policy to use ('rule' | 'llm').

    Heuristic scoring (positive = LLM, negative = Rule):
      +2 if compound (decompose really helps)
      +1 if pronoun (coref helps)
      +1 if open how-to (gap-retry helps)
      -2 if definition / what-is (rewrite hurts)
      -1 if short factoid (no policy work needed)
    """
    f = _features(q)
    score = 0
    if f["has_compound"]:
        score += 2
    if f["has_pronoun"]:
        score += 1
    if f["is_open_howto"]:
        score += 1
    if f["is_definition"]:
        score -= 2
    if f["is_short_factoid"]:
        score -= 1
    return "llm" if score > 0 else "rule"


class RouterPrePolicy(PrePolicy):
    """Dispatches each call to one of two inner PrePolicies.

    Recorded `name` always reflects which inner was used on the latest
    call (handy for trajectory analysis: every step shows which arm fired).
    """

    name = "pre.router"

    def __init__(self, rule: PrePolicy | None = None, llm_policy: PrePolicy | None = None) -> None:
        self._rule = rule or RulePrePolicy()
        self._llm = llm_policy  # may be None if the LLM arm isn't wired yet
        self._last_arm = "rule"

    @property
    def last_arm(self) -> str:
        return self._last_arm

    def decide(self, state: State) -> Action:
        arm = choose(state.question.text)
        # If LLM arm wasn't configured but the router picked it, gracefully
        # fall back to rule rather than crashing.
        if arm == "llm" and self._llm is not None:
            self._last_arm = "llm"
            action = self._llm.decide(state)
        else:
            self._last_arm = "rule"
            action = self._rule.decide(state)
        # tag the action with the chosen arm so trajectory analysis can see it
        action.payload = dict(action.payload or {})
        action.payload.setdefault("_router_arm", self._last_arm)
        return action
