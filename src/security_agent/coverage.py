"""Coverage check.

Produces a structured `Coverage` (score + missing_aspects + low_confidence_claims)
that the Post-Retrieval Policy uses to choose between gap-retry / broaden /
narrow / answer.

This module deliberately uses lexical heuristics rather than LLM calls; an
LLM-based coverage checker can be plugged in by replacing `Coverage.compute`.
"""
from __future__ import annotations

import re

from security_agent.types import Coverage, Document, State


_ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "definition": ("what is", "define", "definition", "meaning"),
    "mitigation": ("mitigate", "mitigation", "fix", "remediation", "patch", "prevent"),
    "exploit": ("exploit", "attack", "vector", "technique"),
    "severity": ("cvss", "severity", "impact", "risk"),
    "detection": ("detect", "detection", "indicator", "ioc"),
    "examples": ("example", "case", "sample"),
}


def _tokens(text: str) -> set[str]:
    return set(w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text) if len(w) > 2)


def _aspect_demand(query: str) -> list[str]:
    q = query.lower()
    demanded: list[str] = []
    for aspect, kws in _ASPECT_KEYWORDS.items():
        if any(k in q for k in kws):
            demanded.append(aspect)
    # if the user asks a generic "how to" / open question, demand mitigation + examples
    if not demanded and any(k in q for k in ("how", "guide", "best practice")):
        demanded = ["mitigation", "examples"]
    return demanded


def _aspect_present(aspect: str, documents: list[Document]) -> bool:
    if aspect not in _ASPECT_KEYWORDS:
        return True
    kws = _ASPECT_KEYWORDS[aspect]
    for d in documents:
        low = d.text.lower()
        if any(k in low for k in kws):
            return True
    return False


def _redundancy(documents: list[Document]) -> float:
    if len(documents) < 2:
        return 0.0
    token_sets = [_tokens(d.text) for d in documents]
    overlaps = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            overlaps.append(len(a & b) / len(a | b))
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def compute_coverage(state: State) -> Coverage:
    """Compute structured coverage from state.documents vs state.question."""
    query = state.effective_query
    docs = state.documents
    if not docs:
        return Coverage(
            score=0.0,
            missing_aspects=_aspect_demand(query) or ["any-evidence"],
            low_confidence_claims=[],
            redundancy=0.0,
            diagnosis="no documents retrieved",
        )

    q_tokens = _tokens(query)
    # lexical coverage of query tokens
    covered = set()
    for d in docs:
        covered |= q_tokens & _tokens(d.text)
    lex_score = len(covered) / max(len(q_tokens), 1) if q_tokens else 0.0

    # aspect coverage
    demanded = _aspect_demand(query)
    missing = [a for a in demanded if not _aspect_present(a, docs)]
    aspect_score = 1.0 - (len(missing) / len(demanded)) if demanded else 1.0

    redundancy = _redundancy(docs)

    # low-confidence: docs with score < median * 0.5
    if docs:
        scores = sorted(d.score for d in docs)
        median = scores[len(scores) // 2]
        threshold = max(0.05, median * 0.5)
        low_conf = [f"[{d.id}]" for d in docs if d.score < threshold]
    else:
        low_conf = []

    score = 0.6 * lex_score + 0.4 * aspect_score
    diagnosis_parts = []
    if missing:
        diagnosis_parts.append(f"missing aspects: {', '.join(missing)}")
    if redundancy > 0.7:
        diagnosis_parts.append(f"high redundancy ({redundancy:.2f})")
    if low_conf:
        diagnosis_parts.append(f"low-confidence claims: {len(low_conf)}")
    diagnosis = "; ".join(diagnosis_parts) or "ok"

    return Coverage(
        score=score,
        missing_aspects=missing,
        low_confidence_claims=low_conf,
        redundancy=redundancy,
        diagnosis=diagnosis,
    )
