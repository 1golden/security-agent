"""Adapter exposing A-Mem (all-in-rag/A-mem-sys) as a MemoryStore.

A-Mem is an optional dependency. Import is safe even if missing; the adapter
only raises when instantiated.

Actual A-Mem API (verified against 0.0.1):
    AgenticMemorySystem(model_name, llm_backend, llm_model, evo_threshold,
                       api_key, sglang_host, sglang_port)
      .memories: dict[str, MemoryNote]          # instance attr
      .add_note(content, time=None, **kwargs) -> str
      .search_agentic(query, k=5) -> list[dict]
      .read(memory_id) -> MemoryNote
      .update(memory_id, **kwargs) -> bool
      .delete(memory_id) -> bool
      .consolidate_memories()                   # reflection / evolution
"""
from __future__ import annotations

import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from security_agent.memory.base import MemoryStore
from security_agent.types import MemoryRecord


class AMemAdapter(MemoryStore):
    """Wrap AgenticMemorySystem.

    A-Mem doesn't natively track sessions, recency, or provider-utility
    feedback — we layer those on top with lightweight side indexes so the
    full MemoryStore contract is honored.
    """

    def __init__(
        self,
        amem_root: str | None = None,
        *,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        model_name: str = "all-MiniLM-L6-v2",
        evo_threshold: int = 100,
        api_key: str | None = None,
    ) -> None:
        if amem_root:
            sys.path.insert(0, str(Path(amem_root)))
        try:
            from agentic_memory.memory_system import (  # type: ignore[import-not-found]
                AgenticMemorySystem,
            )
        except Exception as e:
            raise ImportError(
                "AMemAdapter requires agentic-memory. "
                "Install with `pip install -e <amem_root>`. "
                f"Underlying error: {e}"
            ) from e

        self._sys = AgenticMemorySystem(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
        )
        # side indexes for capabilities A-Mem lacks natively
        self._session_index: dict[str, list[str]] = defaultdict(list)
        self._provider_helps_session: dict[str, Counter[str]] = defaultdict(Counter)
        self._provider_helps_global: Counter[str] = Counter()
        self._session_of: dict[str, str] = {}

    # ----- helpers -----

    @staticmethod
    def _note_to_record(note, score: float = 0.0) -> MemoryRecord:
        """Convert a MemoryNote (or raw dict from search_agentic) into our type."""
        if hasattr(note, "__dict__"):
            d = note.__dict__
        elif isinstance(note, dict):
            d = note
        else:
            d = {"content": str(note)}
        links = d.get("links")
        if isinstance(links, dict):
            link_ids = list(links.keys())
        elif isinstance(links, list):
            link_ids = list(links)
        else:
            link_ids = []
        keywords = d.get("keywords") or []
        tags = d.get("tags") or []
        return MemoryRecord(
            id=d.get("id") or f"a-{uuid.uuid4().hex[:10]}",
            content=d.get("content", ""),
            keywords=list(keywords),
            tags=list(tags),
            context=d.get("context", "") or "",
            links=link_ids,
            score=score,
            metadata={k: v for k, v in d.items()
                      if k not in {"id", "content", "keywords", "tags", "context", "links"}},
        )

    # ----- read -----

    def recent(self, session_id: str, n: int) -> list[MemoryRecord]:
        ids = self._session_index.get(session_id, [])[-n:]
        out: list[MemoryRecord] = []
        for mid in ids:
            note = self._sys.read(mid) if hasattr(self._sys, "read") else None
            if note is None:
                continue
            out.append(self._note_to_record(note))
        return out

    def search(
        self,
        query: str,
        top_k: int,
        session_id: str | None = None,
        providers: list[str] | None = None,
    ) -> list[MemoryRecord]:
        if not query.strip():
            return []
        try:
            raw = self._sys.search_agentic(query, k=top_k) or []
        except Exception:
            return []
        out: list[MemoryRecord] = []
        for r in raw:
            rec = self._note_to_record(r, score=float(r.get("score", 1.0)) if isinstance(r, dict) else 1.0)
            if providers and rec.metadata.get("provider") not in providers:
                continue
            out.append(rec)
        return out[:top_k]

    def neighbors(self, record_id: str, hops: int = 1) -> list[MemoryRecord]:
        seen: set[str] = {record_id}
        frontier: list[str] = [record_id]
        out: list[MemoryRecord] = []
        for _ in range(max(1, hops)):
            new_frontier: list[str] = []
            for nid in frontier:
                note = self._sys.read(nid)
                if note is None:
                    continue
                links = getattr(note, "links", None) or {}
                link_ids = list(links.keys()) if isinstance(links, dict) else list(links)
                for lid in link_ids:
                    if lid in seen:
                        continue
                    seen.add(lid)
                    new_frontier.append(lid)
                    nb = self._sys.read(lid)
                    if nb is not None:
                        out.append(self._note_to_record(nb))
            frontier = new_frontier
            if not frontier:
                break
        return out

    def preferred_providers(self, session_id: str | None = None, top_k: int = 3) -> list[str]:
        counter = (
            self._provider_helps_session.get(session_id, Counter())
            if session_id
            else self._provider_helps_global
        )
        return [p for p, _ in counter.most_common(top_k)]

    # ----- write -----

    def write(self, record: MemoryRecord) -> str:
        kwargs: dict = {}
        if record.keywords:
            kwargs["keywords"] = list(record.keywords)
        if record.tags:
            kwargs["tags"] = list(record.tags)
        if record.context:
            kwargs["context"] = record.context
        # A-Mem's add_note runs LLM-driven analysis if keywords/tags absent; the
        # caller can pre-seed them in the MemoryRecord to save tokens.
        try:
            note_id = self._sys.add_note(content=record.content, **kwargs)
        except Exception as e:
            # Surface the error rather than silently dropping: a failing memory
            # write is something the operator needs to know about.
            raise RuntimeError(f"AMemAdapter.write failed: {e}") from e
        # update side indexes
        sess = record.metadata.get("session_id")
        if sess:
            self._session_index[sess].append(note_id)
            self._session_of[note_id] = sess
        return note_id

    def update(self, record_id: str, patch: dict) -> None:
        try:
            self._sys.update(record_id, **patch)
        except Exception:
            pass  # update is best-effort

    def link(self, src_id: str, dst_id: str) -> None:
        src = self._sys.read(src_id)
        if src is None:
            return
        links = getattr(src, "links", None)
        # MemoryNote.links is a Dict in A-Mem 0.0.1
        if isinstance(links, dict):
            links[dst_id] = links.get(dst_id, 0) + 1
        elif isinstance(links, list):
            if dst_id not in links:
                links.append(dst_id)

    def record_provider_utility(self, session_id: str, provider: str, helped: bool) -> None:
        if not helped or not provider:
            return
        self._provider_helps_session[session_id][provider] += 1
        self._provider_helps_global[provider] += 1

    def trigger_reflection(self, anchor_id: str) -> None:
        # consolidate_memories is the A-Mem-wide reflection / evolution pass.
        # `anchor_id` is unused here because A-Mem's API is global; the caller
        # still passes one so a future per-anchor reflection can be added.
        try:
            self._sys.consolidate_memories()
        except Exception:
            pass
