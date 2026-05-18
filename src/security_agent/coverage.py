"""Coverage check.

Two implementations, plugable into the Pipeline:

- `compute_coverage` (default, heuristic) — lexical with content-pattern
  checks. Fast, deterministic, used by tests and CI. Patched to recognize
  "is a type of …" / "occurs when …" style definitions in the docs rather
  than requiring the literal word "definition" — see Bug #2.
- `LLMCoverage` — wraps an LLMBackend; same Coverage struct. Used when the
  caller wants real semantic coverage analysis (the right thing for
  production / the re-baseline run).

Both produce a Coverage dataclass:
  score, missing_aspects, low_confidence_claims, redundancy, diagnosis
"""
from __future__ import annotations

import json
import re
from typing import Callable, Protocol

from security_agent.generation.llm_generator import LLMBackend
from security_agent.types import Coverage, Document, State


# --------------------------------------------------------------------------
# Aspect taxonomy
# --------------------------------------------------------------------------


# Question-side triggers — does the user ASK for this aspect?
_ASPECT_DEMAND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "definition": ("what is", "what are", "define", "definition", "meaning", "describe"),
    "mitigation": ("mitigate", "mitigation", "fix", "remediation", "patch", "prevent",
                    "defend", "protect", "best practice"),
    "exploit": ("exploit", "attack", "vector", "technique", "abuse"),
    "severity": ("cvss", "severity", "impact", "risk", "score"),
    "detection": ("detect", "detection", "indicator", "ioc", "spot", "find"),
    "examples": ("example", "case", "sample", "instance"),
}

# Document-side: signals that this aspect IS being addressed by a doc, not
# the literal aspect word. Lets the heuristic see "occurs when input is
# concatenated" as a definition without requiring the word "definition".
_ASPECT_PRESENCE_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "definition": (
        re.compile(r"\b(is\s+a\s+(type\s+of|kind\s+of)?|are\s+a\s+(type\s+of|kind\s+of)?)\b", re.I),
        re.compile(r"\b(refers\s+to|occurs\s+when|happens\s+when|defined\s+as)\b", re.I),
        re.compile(r"\b(what\s+is|describes?|definition|meaning)\b", re.I),
    ),
    "mitigation": (
        re.compile(r"\b(prevent|mitigate|fix|patch|remediation|defend|protect)\w*\b", re.I),
        re.compile(r"\b(use|enforce|enable|configure|apply)\s+\w+\s+to\s+(prevent|avoid|stop)\b", re.I),
        re.compile(r"\b(parameteriz|whitelist|allowlist|sanitize|validate)\w*\b", re.I),
        re.compile(r"\b(least\s+privilege|defense\s+in\s+depth|secure\s+by\s+default)\b", re.I),
    ),
    "exploit": (
        re.compile(r"\b(exploit|attack|payload|injection|vector|technique)\w*\b", re.I),
        re.compile(r"\b(attackers?\s+(can|may|might)|adversar(y|ies)|threat\s+actor)\b", re.I),
    ),
    "severity": (
        re.compile(r"\b(cvss|severity|critical|high|medium|low)\b", re.I),
        re.compile(r"\b(impact|score|risk\s+rating)\b", re.I),
    ),
    "detection": (
        re.compile(r"\b(detect|detection|monitor|log|alert|ioc|indicator)\w*\b", re.I),
        re.compile(r"\b(siem|edr|anomaly|signature)\b", re.I),
    ),
    "examples": (
        re.compile(r"\b(for\s+example|e\.g\.|such\s+as|sample|case\s+study)\b", re.I),
        re.compile(r"```|\bexample[s]?:\b", re.I),
    ),
}


def _tokens(text: str) -> set[str]:
    return set(w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text) if len(w) > 2)


def _aspect_demand(query: str) -> list[str]:
    q = query.lower()
    demanded: list[str] = []
    for aspect, kws in _ASPECT_DEMAND_KEYWORDS.items():
        if any(k in q for k in kws):
            demanded.append(aspect)
    if not demanded and any(k in q for k in ("how", "guide")):
        demanded = ["mitigation", "examples"]
    return demanded


def _aspect_present(aspect: str, documents: list[Document]) -> bool:
    """Bug #2: presence is detected via CONTENT PATTERNS, not literal keywords.

    A doc that explains "SQL injection occurs when input is concatenated into a
    query" satisfies `definition` even though it never uses the word
    "definition" itself.
    """
    if aspect not in _ASPECT_DEMAND_KEYWORDS:
        return True
    patterns = _ASPECT_PRESENCE_PATTERNS.get(aspect, ())
    legacy_kws = _ASPECT_DEMAND_KEYWORDS[aspect]
    for d in documents:
        text = d.text or ""
        if not text:
            continue
        # fast path: literal keyword (still useful)
        low = text.lower()
        if any(k in low for k in legacy_kws):
            return True
        # content pattern check
        if any(p.search(text) for p in patterns):
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
    """Heuristic coverage. Pluggable: replace via Pipeline.coverage_fn."""
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
    covered = set()
    for d in docs:
        covered |= q_tokens & _tokens(d.text)
    lex_score = len(covered) / max(len(q_tokens), 1) if q_tokens else 0.0

    demanded = _aspect_demand(query)
    missing = [a for a in demanded if not _aspect_present(a, docs)]
    aspect_score = 1.0 - (len(missing) / len(demanded)) if demanded else 1.0

    redundancy = _redundancy(docs)

    scores = sorted(d.score for d in docs)
    median = scores[len(scores) // 2]
    threshold = max(0.05, median * 0.5)
    low_conf = [f"[{d.id}]" for d in docs if d.score < threshold]

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


# --------------------------------------------------------------------------
# CoverageFn protocol + LLM-based implementation
# --------------------------------------------------------------------------


class CoverageFn(Protocol):
    def __call__(self, state: State) -> Coverage: ...


_LLM_COV_SYSTEM = (
    "You are a retrieval-quality auditor for a security-domain RAG. Given a "
    "question and the retrieved evidence, judge how well the evidence "
    "actually addresses the question.\n"
    "Score (0..1):\n"
    "  - score: overall how well the evidence supports a good answer\n"
    "  - redundancy: how much the docs duplicate each other (0=diverse, 1=all same)\n"
    "Identify missing aspects from this taxonomy: "
    "definition, mitigation, exploit, severity, detection, examples. "
    "Identify any retrieved chunks whose relevance is dubious "
    "(`low_confidence_claims` as list of doc ids).\n"
    "Reply JSON ONLY: "
    '{"score": <0..1>, "redundancy": <0..1>, "missing_aspects": [...], '
    '"low_confidence_claims": [...], "diagnosis": "<one line>"}'
)

_ALLOWED_ASPECTS = set(_ASPECT_DEMAND_KEYWORDS.keys())


class LLMCoverage:
    """LLM-driven coverage check. Returns the same Coverage struct."""

    def __init__(self, llm: LLMBackend, max_evidence_chars: int = 6000) -> None:
        self._llm = llm
        self._max_ev = max_evidence_chars

    def __call__(self, state: State) -> Coverage:
        docs = state.documents
        if not docs:
            return Coverage(
                score=0.0,
                missing_aspects=_aspect_demand(state.effective_query) or ["any-evidence"],
                low_confidence_claims=[],
                redundancy=0.0,
                diagnosis="no documents retrieved",
            )
        ev_lines = [f"[{d.id}] ({d.source or '?'}) {d.text[:600]}" for d in docs]
        ev_block = "\n".join(ev_lines)
        if len(ev_block) > self._max_ev:
            ev_block = ev_block[: self._max_ev] + "\n... (truncated)"

        user = (
            f"Question: {state.effective_query}\n\n"
            f"Retrieved evidence ({len(docs)} chunks):\n{ev_block}"
        )
        try:
            data = self._llm.json(_LLM_COV_SYSTEM, user)
        except Exception:
            data = {}

        def _f(key: str, default: float = 0.5) -> float:
            v = data.get(key, default)
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return default

        missing_raw = data.get("missing_aspects") or []
        if not isinstance(missing_raw, list):
            missing_raw = []
        missing = [str(a).strip().lower() for a in missing_raw if str(a).strip()]
        missing = [a for a in missing if a in _ALLOWED_ASPECTS]

        low_conf_raw = data.get("low_confidence_claims") or []
        if not isinstance(low_conf_raw, list):
            low_conf_raw = []
        # normalize to [id] form so the FILTER post-action can strip them
        low_conf = []
        for x in low_conf_raw:
            s = str(x).strip()
            if not s:
                continue
            if not (s.startswith("[") and s.endswith("]")):
                s = f"[{s.strip('[]')}]"
            low_conf.append(s)

        return Coverage(
            score=_f("score", 0.5),
            missing_aspects=missing,
            low_confidence_claims=low_conf,
            redundancy=_f("redundancy", _redundancy(docs)),
            diagnosis=str(data.get("diagnosis", ""))[:200] or "llm-graded",
        )


def build_coverage(kind: str = "heuristic", **kwargs) -> CoverageFn:
    kind = (kind or "heuristic").lower()
    if kind == "heuristic":
        return compute_coverage
    if kind == "llm":
        llm = kwargs.get("llm")
        if llm is None:
            raise ValueError("LLM coverage requires `llm=<LLMBackend>`")
        return LLMCoverage(
            llm=llm,
            max_evidence_chars=kwargs.get("max_evidence_chars", 6000),
        )
    raise ValueError(f"Unknown coverage kind: {kind}")
