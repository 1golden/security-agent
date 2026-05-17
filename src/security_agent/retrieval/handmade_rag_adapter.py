"""Adapter exposing all-in-rag/code/handmade_rag as a Retriever.

handmade_rag is optional. Import is safe even if absent; the adapter only
raises when instantiated.

Actual handmade_rag API (verified against the local source):
    HandmadeRAG(config: RagConfig | None = None)
      .load() -> None               # no args, uses config.index_dir
      .search(question, filters) -> dict with key "results"
        each result: {chunk_id, provider, title, section_title, score,
                      source, doc_id, source_span: {text, ...}}
"""
from __future__ import annotations

import sys
from pathlib import Path

from security_agent.retrieval.base import Retriever
from security_agent.types import Document


class HandmadeRagAdapter(Retriever):
    def __init__(
        self,
        root: str | None = None,
        index_dir: str | None = None,
    ) -> None:
        # ensure the package is on path if it's a checkout (non-installed)
        if root and not _is_installed("handmade_rag"):
            sys.path.insert(0, str(Path(root) / "src"))
        try:
            from handmade_rag.pipeline import HandmadeRAG  # type: ignore[import-not-found]
            from handmade_rag.config import RagConfig
        except Exception as e:
            raise ImportError(
                f"HandmadeRagAdapter requires handmade_rag. "
                f"Install with `pip install -e <root>` or pass root=<dir>. ({e})"
            ) from e

        cfg = RagConfig()
        if index_dir:
            cfg.index_dir = Path(index_dir)
        else:
            # try the conventional artifact path
            if root:
                candidate = Path(root) / "artifacts" / "security_index"
                if candidate.exists():
                    cfg.index_dir = candidate
        self._rag = HandmadeRAG(config=cfg)
        try:
            self._rag.load()
        except Exception as e:
            raise RuntimeError(
                f"HandmadeRAG.load() failed for index_dir={cfg.index_dir}: {e}"
            ) from e

    def retrieve(
        self,
        query: str,
        top_k: int,
        providers: list[str] | None = None,
        expansion_terms: list[str] | None = None,
    ) -> list[Document]:
        expanded = query
        if expansion_terms:
            expanded = f"{query} {' '.join(expansion_terms)}".strip()

        # handmade_rag's search signature is (question, filters); top_k is
        # controlled by config.retrieval.rerank_top_k. We do client-side
        # top_k truncation to honor our caller's intent.
        filters: dict = {}
        if providers:
            # handmade_rag supports a "provider" filter via its filters dict
            filters["provider"] = providers if len(providers) > 1 else providers[0]

        try:
            resp = self._rag.search(expanded, filters=filters or None)
        except Exception as e:
            raise RuntimeError(f"HandmadeRAG.search failed: {e}") from e

        results = (resp or {}).get("results") or []
        out: list[Document] = []
        for r in results:
            source_span = r.get("source_span") or {}
            text = source_span.get("text") or ""
            if not text:
                # fall back to title + section_title so the LLM at least has
                # something to ground on
                text = " ".join(
                    [r.get("title") or "", r.get("section_title") or ""]
                ).strip()
            d = Document(
                id=str(r.get("chunk_id") or r.get("doc_id") or len(out)),
                text=text,
                score=float(r.get("score") or 0.0),
                source=r.get("provider") or r.get("source"),
                metadata={
                    "title": r.get("title"),
                    "section_title": r.get("section_title"),
                    "collection": r.get("collection"),
                    "category": r.get("category"),
                    "doc_id": r.get("doc_id"),
                },
            )
            out.append(d)
        return out[:top_k]


def _is_installed(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False
