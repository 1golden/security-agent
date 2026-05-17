"""In-process memory store. Zero dependencies, suitable for tests and demos."""
from __future__ import annotations

import re
import time
import uuid
from collections import Counter, defaultdict, deque

from security_agent.memory.base import MemoryStore
from security_agent.types import MemoryRecord


def _tokens(text: str) -> set[str]:
    return set(w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text) if len(w) > 2)


class InMemoryStore(MemoryStore):
    """Naive keyword-overlap retrieval. Good enough for the rule baseline."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._by_session: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=64))
        self._provider_helps: dict[str, Counter[str]] = defaultdict(Counter)
        self._global_provider_helps: Counter[str] = Counter()

    # ----- read -----

    def recent(self, session_id: str, n: int) -> list[MemoryRecord]:
        ids = list(self._by_session.get(session_id, ()))[-n:]
        return [self._records[i] for i in ids if i in self._records]

    def search(
        self,
        query: str,
        top_k: int,
        session_id: str | None = None,
        providers: list[str] | None = None,
    ) -> list[MemoryRecord]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        scored: list[tuple[float, MemoryRecord]] = []
        for rec in self._records.values():
            if providers:
                src = rec.metadata.get("provider")
                if src and src not in providers:
                    continue
            r_tokens = _tokens(rec.content) | set(t.lower() for t in rec.keywords + rec.tags)
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens) / max(len(q_tokens), 1)
            if overlap > 0:
                rec.score = overlap
                scored.append((overlap, rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def neighbors(self, record_id: str, hops: int = 1) -> list[MemoryRecord]:
        seen: set[str] = {record_id}
        frontier: list[str] = [record_id]
        out: list[MemoryRecord] = []
        for _ in range(hops):
            new_frontier: list[str] = []
            for rid in frontier:
                rec = self._records.get(rid)
                if not rec:
                    continue
                for link in rec.links:
                    if link in seen or link not in self._records:
                        continue
                    seen.add(link)
                    new_frontier.append(link)
                    out.append(self._records[link])
            frontier = new_frontier
            if not frontier:
                break
        return out

    def preferred_providers(self, session_id: str | None = None, top_k: int = 3) -> list[str]:
        counter = (
            self._provider_helps.get(session_id, Counter())
            if session_id
            else self._global_provider_helps
        )
        return [p for p, _ in counter.most_common(top_k)]

    # ----- write -----

    def write(self, record: MemoryRecord) -> str:
        if not record.id:
            record.id = f"m-{uuid.uuid4().hex[:10]}"
        record.ingestion_time = record.ingestion_time or time.time()
        self._records[record.id] = record
        sess = record.metadata.get("session_id")
        if sess:
            self._by_session[sess].append(record.id)
        return record.id

    def update(self, record_id: str, patch: dict) -> None:
        rec = self._records.get(record_id)
        if not rec:
            return
        for k, v in patch.items():
            if hasattr(rec, k):
                setattr(rec, k, v)
            else:
                rec.metadata[k] = v

    def link(self, src_id: str, dst_id: str) -> None:
        src = self._records.get(src_id)
        if src and dst_id not in src.links:
            src.links.append(dst_id)

    def record_provider_utility(
        self, session_id: str, provider: str, helped: bool
    ) -> None:
        if not helped:
            return
        self._provider_helps[session_id][provider] += 1
        self._global_provider_helps[provider] += 1

    def trigger_reflection(self, anchor_id: str) -> None:
        """Light-weight reflection: re-link an anchor to its top keyword neighbors.

        A real A-Mem reflection would call an LLM to refine keywords/tags. Here
        we just symmetrize links between the anchor and items sharing >=2
        keywords, which is enough for the rule baseline + tests.
        """
        anchor = self._records.get(anchor_id)
        if not anchor:
            return
        a_kw = set(k.lower() for k in anchor.keywords)
        if not a_kw:
            return
        for rid, rec in self._records.items():
            if rid == anchor_id:
                continue
            overlap = len(a_kw & set(k.lower() for k in rec.keywords))
            if overlap >= 2:
                self.link(anchor_id, rid)
                self.link(rid, anchor_id)
