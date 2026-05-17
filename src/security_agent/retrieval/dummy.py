"""Dummy retriever backed by a small in-process corpus.

Useful for tests, demos, and CI where the real index is not available. The
corpus is keyword-overlap scored; provider filtering is honored.
"""
from __future__ import annotations

import re

from security_agent.retrieval.base import Retriever
from security_agent.types import Document


def _tokens(text: str) -> set[str]:
    return set(w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text) if len(w) > 2)


_DEFAULT_CORPUS: list[tuple[str, str, str]] = [
    (
        "owasp-a01",
        "owasp",
        "OWASP A01:2021 Broken Access Control: enforce least privilege and deny by default.",
    ),
    (
        "owasp-a03",
        "owasp",
        "OWASP A03:2021 Injection: parameterize queries, validate inputs server-side.",
    ),
    (
        "nist-800-53-ac",
        "nist",
        "NIST 800-53 AC family covers access control policies and procedures.",
    ),
    (
        "mitre-t1078",
        "mitre",
        "MITRE ATT&CK T1078 Valid Accounts: adversaries may use existing credentials.",
    ),
    (
        "cisa-known-exploited",
        "cisa",
        "CISA KEV catalog lists vulnerabilities known to be actively exploited.",
    ),
    (
        "cve-2024-1234",
        "nvd",
        "Sample CVE-2024-1234 describes a SQL injection in webapp foobar 1.0.",
    ),
]


class DummyRetriever(Retriever):
    def __init__(self, corpus: list[tuple[str, str, str]] | None = None) -> None:
        self._corpus = corpus or _DEFAULT_CORPUS

    def retrieve(
        self,
        query: str,
        top_k: int,
        providers: list[str] | None = None,
        expansion_terms: list[str] | None = None,
    ) -> list[Document]:
        q_tokens = _tokens(query)
        if expansion_terms:
            q_tokens |= _tokens(" ".join(expansion_terms))
        if not q_tokens:
            return []
        scored: list[tuple[float, Document]] = []
        for doc_id, source, text in self._corpus:
            if providers and source not in providers:
                continue
            d_tokens = _tokens(text)
            overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
            if overlap > 0:
                scored.append((overlap, Document(id=doc_id, text=text, score=overlap, source=source)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in scored[:top_k]]
