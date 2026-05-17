"""Memory Expansion: heavyweight read used AFTER the planner decides.

Combines:
- vector / keyword search on rewritten query + expansion terms
- KG multi-hop seeded from probe seed nodes
- provider-filtered if Pre-Retrieval Policy chose CHOOSE_PROVIDER
"""
from __future__ import annotations

from security_agent.memory.base import MemoryStore
from security_agent.types import MemoryRecord, State


class MemoryExpansion:
    def __init__(self, store: MemoryStore, top_k: int = 5, kg_hops: int = 2) -> None:
        self._store = store
        self._top_k = top_k
        self._kg_hops = kg_hops

    def __call__(self, state: State) -> list[MemoryRecord]:
        queries: list[str] = []
        queries.append(state.effective_query)
        queries.extend(state.subqueries)
        if state.expansion_terms:
            queries.append(" ".join(state.expansion_terms))

        providers = state.provider_filter or None

        seen: dict[str, MemoryRecord] = {}
        for q in queries:
            hits = self._store.search(
                q,
                top_k=self._top_k,
                session_id=state.question.session_id,
                providers=providers,
            )
            for h in hits:
                if h.id not in seen or h.score > seen[h.id].score:
                    seen[h.id] = h

        # multi-hop from probe seeds: search for each seed, then expand via links
        for seed in state.probe.kg_seed_nodes[:3]:
            seed_hits = self._store.search(seed, top_k=2, providers=providers)
            for h in seed_hits:
                seen.setdefault(h.id, h)
                for nb in self._store.neighbors(h.id, hops=self._kg_hops):
                    seen.setdefault(nb.id, nb)

        return sorted(seen.values(), key=lambda r: r.score, reverse=True)[: self._top_k * 2]
