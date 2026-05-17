"""Fast Memory Probe.

Cheap signals only — no embedding, no LLM. Designed to give the Pre-Retrieval
Policy enough hints to plan without paying for full memory expansion.
"""
from __future__ import annotations

import re

from security_agent.memory.base import MemoryStore
from security_agent.types import ProbeSignals, Question


PRONOUNS = {"it", "they", "this", "that", "these", "those", "he", "she", "him", "her"}
FILTER_RE = re.compile(
    r"(?:in|on|using|for)\s+(owasp|nist|cisa|mitre|att&ck|nvd|cve|gb/?\s*\d+)",
    re.IGNORECASE,
)


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

        # coref hints: if a pronoun appears in the question, look in the last
        # turn for a capitalized noun and map it.
        coref: dict[str, str] = {}
        q_tokens = re.findall(r"[A-Za-z]+", question.text.lower())
        if any(p in q_tokens for p in PRONOUNS) and recent:
            last = recent[-1].content
            cand = re.findall(r"\b([A-Z][a-zA-Z0-9]{2,})\b", last)
            if cand:
                for p in PRONOUNS:
                    if p in q_tokens:
                        coref[p] = cand[0]
                        break

        # KG seed nodes: pull entity-like keywords from recent records
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
