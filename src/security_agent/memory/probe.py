"""Fast Memory Probe.

Cheap signals only — no embedding, no LLM. Designed to give the Pre-Retrieval
Policy enough hints to plan without paying for full memory expansion.
"""
from __future__ import annotations

import re
from collections import Counter

from security_agent.memory.base import MemoryStore
from security_agent.types import ProbeSignals, Question


PRONOUNS = {"it", "they", "this", "that", "these", "those", "he", "she", "him", "her"}

# Words that are commonly capitalized at sentence-start but are NOT entities.
# Picking these as coref targets corrupts the rewritten query (Bug #5).
COREF_STOPLIST = {
    "Based", "The", "This", "That", "These", "Those", "Note",
    "However", "Therefore", "Although", "While", "When", "After",
    "Before", "Since", "First", "Second", "Third", "Then", "But",
    "And", "Or", "Also", "Both", "Either", "Neither", "I", "We",
    "You", "He", "She", "It", "They", "Their", "Our", "Your",
    "Here", "There", "Now", "Today", "Yes", "No", "Maybe",
    "Generally", "Typically", "Specifically", "Importantly",
    "Otherwise", "Meanwhile", "Furthermore", "Moreover",
    "Answer", "Question", "Reply", "Response", "Evidence",
}

FILTER_RE = re.compile(
    r"(?:in|on|using|for)\s+(owasp|nist|cisa|mitre|att&ck|nvd|cve|gb/?\s*\d+)",
    re.IGNORECASE,
)


def _score_coref_candidate(token: str) -> int:
    """Bug #5 fix: prefer acronyms (e.g. OWASP, NIST, SQL) and multi-cap
    tokens; penalize stop-list words and single-cap tokens.

    Higher is better. Returns 0 for disqualified candidates.
    """
    if token in COREF_STOPLIST:
        return 0
    if len(token) < 3:
        return 0
    cap_count = sum(1 for c in token if c.isupper())
    if cap_count >= 3:
        return 10 + cap_count  # OWASP, NIST, CVE, SQL, MITRE
    if "_" in token or "-" in token:
        return 8
    if any(c.isdigit() for c in token):
        return 7  # A03, 2023, T1078
    # plain capitalized word — accept but low priority
    return 1


class FastMemoryProbe:
    """Build a ProbeSignals from question + memory in O(memory recent-window)."""

    def __init__(self, store: MemoryStore, recent_n: int = 4) -> None:
        self._store = store
        self._recent_n = recent_n

    def __call__(self, question: Question) -> ProbeSignals:
        recent = self._store.recent(question.session_id, self._recent_n)

        # explicit filters: scan question text for known provider mentions
        explicit_filters: dict[str, object] = {}
        m = FILTER_RE.search(question.text)
        if m:
            explicit_filters["provider"] = m.group(1).lower().replace(" ", "").replace("/", "")

        coref: dict[str, str] = self._resolve_coref(question, recent)

        seeds: list[str] = []
        for rec in recent:
            for kw in rec.keywords[:3]:
                if kw not in seeds:
                    seeds.append(kw)

        return ProbeSignals(
            recent_turns=[r.content for r in recent],
            explicit_filters=explicit_filters,
            coref_hints=coref,
            kg_seed_nodes=seeds,
            preferred_providers=self._store.preferred_providers(
                question.session_id, top_k=3
            ),
        )

    @staticmethod
    def _resolve_coref(question: Question, recent: list) -> dict[str, str]:
        """Pick the best entity in the prior turn to bind a pronoun to.

        Strategy (Bug #5 fix):
        1. Collect ALL capitalized candidates across the prior turn.
        2. Score each by acronym-ness; filter the stop-list.
        3. Prefer tokens that appeared multiple times (likely the topic).
        4. Only emit a hint if a clear winner exists (score >= 7).
        """
        if not recent:
            return {}
        q_tokens = re.findall(r"[A-Za-z]+", question.text.lower())
        pronouns_here = [p for p in PRONOUNS if p in q_tokens]
        if not pronouns_here:
            return {}

        last = recent[-1].content
        raw_cands = re.findall(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b", last)
        if not raw_cands:
            return {}

        freq = Counter(raw_cands)
        scored: list[tuple[int, str]] = []
        for token, count in freq.items():
            base = _score_coref_candidate(token)
            if base == 0:
                continue
            scored.append((base + count, token))
        if not scored:
            return {}
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_token = scored[0]
        # only emit if confidence is reasonable — drop weak picks
        if best_score < 7:
            return {}
        # bind every pronoun in the question to the best entity
        return {p: best_token for p in pronouns_here}
