"""Adapter that exposes all-in-rag/code/handmade_rag as a Retriever.

handmade_rag is optional. If unavailable, importing this module is safe; the
adapter only raises when instantiated.
"""
from __future__ import annotations

import sys
from pathlib import Path

from security_agent.retrieval.base import Retriever
from security_agent.types import Document


class HandmadeRagAdapter(Retriever):
    def __init__(self, root: str, index_dir: str | None = None) -> None:
        sys.path.insert(0, str(Path(root)))
        try:
            from handmade_rag.pipeline import HandmadeRAG  # type: ignore[import-not-found]
        except Exception as e:
            raise ImportError(
                f"HandmadeRagAdapter requires handmade_rag at {root!r} ({e})"
            ) from e
        self._rag = HandmadeRAG()
        if index_dir:
            self._rag.load(index_dir)
        else:
            # try the standard artifact path
            default = Path(root) / "artifacts" / "security_index"
            if default.exists():
                self._rag.load(str(default))

    def retrieve(
        self,
        query: str,
        top_k: int,
        providers: list[str] | None = None,
        expansion_terms: list[str] | None = None,
    ) -> list[Document]:
        expanded = query
        if expansion_terms:
            expanded = query + " " + " ".join(expansion_terms)
        raw = self._rag.search(expanded, top_k=top_k)  # type: ignore[attr-defined]
        out: list[Document] = []
        for r in raw or []:
            d = Document(
                id=str(r.get("id") or r.get("chunk_id") or len(out)),
                text=r.get("text", "") or r.get("retrieval_text", ""),
                score=float(r.get("score", 0.0)),
                source=r.get("provider") or r.get("source"),
                metadata={k: v for k, v in r.items() if k not in {"id", "text", "score"}},
            )
            if providers and d.source not in providers:
                continue
            out.append(d)
        return out[:top_k]
