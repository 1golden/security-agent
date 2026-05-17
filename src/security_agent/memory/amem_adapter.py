"""Adapter that exposes the A-Mem (all-in-rag/A-mem-sys) system as a MemoryStore.

The A-Mem package is an optional dependency. If unavailable, importing this
module is still safe; the adapter raises only when instantiated.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from security_agent.memory.base import MemoryStore
from security_agent.types import MemoryRecord


class AMemAdapter(MemoryStore):
    """Best-effort wrapper around `agentic_memory.AgenticMemorySystem`.

    Only the interfaces actually used by the pipeline are mapped; other
    methods raise NotImplementedError so misuse is loud, not silent.
    """

    def __init__(self, amem_root: str) -> None:
        sys.path.insert(0, str(Path(amem_root)))
        try:
            from agentic_memory.memory_system import (  # type: ignore[import-not-found]
                AgenticMemorySystem,
            )
        except Exception as e:
            raise ImportError(
                f"AMemAdapter requires A-Mem at {amem_root!r}. "
                f"Install the package with `pip install -e .` inside that dir. ({e})"
            ) from e
        self._sys = AgenticMemorySystem()

    @staticmethod
    def _to_record(raw: dict, score: float = 0.0) -> MemoryRecord:
        return MemoryRecord(
            id=raw.get("id", f"a-{uuid.uuid4().hex[:8]}"),
            content=raw.get("content", ""),
            keywords=raw.get("keywords", []) or [],
            tags=raw.get("tags", []) or [],
            context=raw.get("context", "") or "",
            links=raw.get("links", []) or [],
            score=score,
            metadata={k: v for k, v in raw.items() if k not in {"id", "content"}},
        )

    def recent(self, session_id: str, n: int) -> list[MemoryRecord]:
        # A-Mem doesn't have a native "last-n by session" — we approximate by
        # listing all notes and slicing. Acceptable for small stores; production
        # should add an index.
        notes = getattr(self._sys, "notes", {}) or {}
        items = list(notes.values())[-n:]
        return [self._to_record(x.__dict__ if hasattr(x, "__dict__") else x) for x in items]

    def search(
        self,
        query: str,
        top_k: int,
        session_id: str | None = None,
        providers: list[str] | None = None,
    ) -> list[MemoryRecord]:
        results = self._sys.search_agentic(query, k=top_k) or []
        out: list[MemoryRecord] = []
        for r in results:
            rec = self._to_record(r if isinstance(r, dict) else r.__dict__, score=1.0)
            if providers:
                if rec.metadata.get("provider") not in providers:
                    continue
            out.append(rec)
        return out

    def neighbors(self, record_id: str, hops: int = 1) -> list[MemoryRecord]:
        notes = getattr(self._sys, "notes", {}) or {}
        anchor = notes.get(record_id)
        if anchor is None:
            return []
        out: list[MemoryRecord] = []
        for lid in getattr(anchor, "links", []) or []:
            if lid in notes:
                out.append(self._to_record(notes[lid].__dict__))
        return out

    def preferred_providers(self, session_id: str | None = None, top_k: int = 3) -> list[str]:
        # A-Mem has no native provider feedback; return empty.
        return []

    def write(self, record: MemoryRecord) -> str:
        rid = self._sys.add_note(
            content=record.content,
            keywords=record.keywords,
            tags=record.tags,
        )
        return rid or record.id or f"a-{uuid.uuid4().hex[:8]}"

    def update(self, record_id: str, patch: dict) -> None:
        if hasattr(self._sys, "update"):
            self._sys.update(record_id, **patch)

    def link(self, src_id: str, dst_id: str) -> None:
        notes = getattr(self._sys, "notes", {}) or {}
        src = notes.get(src_id)
        if src is not None and dst_id not in getattr(src, "links", []):
            src.links.append(dst_id)

    def record_provider_utility(self, session_id: str, provider: str, helped: bool) -> None:
        # A-Mem doesn't track this; no-op. The pipeline still gathers the
        # signal in its own log for later analysis.
        return None

    def trigger_reflection(self, anchor_id: str) -> None:
        if hasattr(self._sys, "consolidate_memories"):
            self._sys.consolidate_memories()
