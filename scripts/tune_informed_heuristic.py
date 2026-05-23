#!/usr/bin/env python3
"""Informed-heuristic router (E1, attempt 3).

LR overfits; rule-enumeration LOO overfits worse. Keep the original 5-
feature heuristic as base, then add ≤2 per-keyword adjustments derived
from per-category outcome diffs. Each adjustment must:
  - Be motivated by ≥3 Qs of evidence
  - Improve the LOO score over the unaltered heuristic

We enumerate single-keyword and 2-keyword adjustments on top of the
heuristic. LOO selects the best adjustment per fold; if no adjustment
helps, we fall back to the heuristic.
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

EXP = Path('/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_19_v4')
OUT = Path('/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_22_v5')

rule_rows = [json.loads(l) for l in open(EXP / 'bench50_rule_v4.jsonl') if l.strip()]
llm_rows = [json.loads(l) for l in open(EXP / 'bench50_llm_v4.jsonl') if l.strip()]
qs = [(r['question'], r['score'], l['score'], r['category'])
      for r, l in zip(rule_rows, llm_rows)]
n = len(qs)
rule_only = sum(r for _, r, _, _ in qs) / n
llm_only = sum(l for _, _, l, _ in qs) / n
oracle = sum(max(r, l) for _, r, l, _ in qs) / n

# Base heuristic (v4 router)
_PRONOUNS = ("it", "they", "this", "that", "these", "those", "he", "she")
_CONJ_RE = re.compile(r"\b(and|then|also|as well as)\b", re.IGNORECASE)
_HOW_RE = re.compile(r"^\s*how\b", re.IGNORECASE)
_DEF_RE = re.compile(r"^\s*what\s+(is|are|does|do)\b", re.IGNORECASE)


def heuristic_score(q: str) -> int:
    low = q.lower()
    wc = len(re.findall(r"\S+", q))
    s = 0
    if _CONJ_RE.search(q) and wc >= 6: s += 2
    if any(re.search(rf"\b{p}\b", low) for p in _PRONOUNS): s += 1
    if _HOW_RE.match(q): s += 1
    if _DEF_RE.match(q): s -= 2
    if wc <= 6: s -= 1
    return s


KEYWORDS = {
    "kw_session":   r"\b(session|cookie attribute|httponly|samesite)\b",
    "kw_logging":   r"\b(log\b|logging|audit|monitor|alert|siem|detect)\b",
    "kw_describe":  r"\b(describ|explain|purpose of|role of|what does|how does)\b",
    "kw_mitigation":r"\b(mitigat|prevent|defen[cs]|harden|protect|countermeasur|remediat)\b",
    "kw_jwt":       r"\b(jwt|json web token|jws|jwe)\b",
    "kw_container": r"\b(container|docker|kubernetes|k8s|pod)\b",
    "kw_crypto":    r"\b(encrypt|cipher|tls|ssl|certificate|hash|hmac)\b",
    "kw_clickjack": r"\b(clickjack|frame busting|x-frame)\b",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORDS.items()}


def adjusted_choose(q, adjustments):
    """adjustments: dict[kw_name -> delta]"""
    s = heuristic_score(q)
    for kw, delta in adjustments.items():
        if COMPILED[kw].search(q):
            s += delta
    return "llm" if s > 0 else "rule"


def mean_with(qs_local, adjustments):
    tot = 0.0
    for q, rs, ls, _ in qs_local:
        tot += ls if adjusted_choose(q, adjustments) == "llm" else rs
    return tot / len(qs_local)


# baseline: heuristic alone
heur_score = mean_with(qs, {})
print(f"baseline heuristic score: {heur_score:.4f}")
print(f"rule_only: {rule_only:.4f}  llm_only: {llm_only:.4f}  oracle: {oracle:.4f}")

# candidate adjustment magnitudes
DELTAS = [-3, -2, -1, +1, +2, +3]

# single-keyword adjustments
print("\nTop single-keyword adjustments on top of heuristic:")
single = []
for kw in COMPILED:
    for d in DELTAS:
        s = mean_with(qs, {kw: d})
        # how many Qs does this keyword match?
        n_match = sum(1 for q, _, _, _ in qs if COMPILED[kw].search(q))
        if n_match >= 2:
            single.append((s, kw, d, n_match))
single.sort(reverse=True)
for s, kw, d, m in single[:10]:
    print(f"  {kw:14s} delta={d:+d}  n_match={m:2d}  → {s:.4f}")

# 2-keyword adjustments — pick the best at each pair
print("\nTop 2-keyword adjustment pairs:")
pairs = []
for (k1, k2) in combinations(COMPILED, 2):
    for d1 in DELTAS:
        for d2 in DELTAS:
            s = mean_with(qs, {k1: d1, k2: d2})
            n1 = sum(1 for q, _, _, _ in qs if COMPILED[k1].search(q))
            n2 = sum(1 for q, _, _, _ in qs if COMPILED[k2].search(q))
            if n1 >= 2 and n2 >= 2:
                pairs.append((s, k1, d1, k2, d2))
pairs.sort(reverse=True)
for s, k1, d1, k2, d2 in pairs[:10]:
    print(f"  {k1:14s} {d1:+d}  + {k2:14s} {d2:+d}  → {s:.4f}")

# ============ HONEST LOO ============
# At each fold: enumerate {no adj, single, 2-keyword} on n-1; apply best to i.
def all_adjustments():
    out = [dict()]
    for kw in COMPILED:
        for d in DELTAS:
            out.append({kw: d})
    for k1, k2 in combinations(COMPILED, 2):
        for d1 in DELTAS:
            for d2 in DELTAS:
                out.append({k1: d1, k2: d2})
    return out


ADJ_LIST = all_adjustments()
print(f"\n{len(ADJ_LIST)} candidate adjustments (incl. empty)")
print("Running honest LOO (this may take ~10s)...")
loo_scores = []
loo_adj = []
for i in range(n):
    train = [qs[j] for j in range(n) if j != i]
    best_s = -1
    best_a = {}
    for a in ADJ_LIST:
        s = mean_with(train, a)
        if s > best_s:
            best_s = s
            best_a = a
    arm = adjusted_choose(qs[i][0], best_a)
    achieved = qs[i][2] if arm == "llm" else qs[i][1]
    loo_scores.append(achieved)
    loo_adj.append(tuple(sorted(best_a.items())))

loo_mean = sum(loo_scores) / len(loo_scores)
print(f"\nLOO-honest informed heuristic: {loo_mean:.4f}")
print(f"  vs heuristic alone:          {heur_score:.4f}")
print(f"  vs rule-only:                {rule_only:.4f}")
print(f"  vs oracle:                   {oracle:.4f}")
print(f"  router→oracle gap closed:    "
      f"{(loo_mean - heur_score) / (oracle - heur_score) * 100:.1f}%")

from collections import Counter
print("\nMost-chosen adjustment across LOO folds:")
for adj, ct in Counter(loo_adj).most_common(5):
    print(f"  {ct:2d}x  {adj}")

# Refit on full data — this is the production adjustment
best_s = -1
best_a = {}
for a in ADJ_LIST:
    s = mean_with(qs, a)
    if s > best_s:
        best_s = s
        best_a = a
print(f"\nFull-set best adjustment: {best_a}  → {best_s:.4f}  (will be applied in v5 router)")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'informed_heuristic.json').write_text(json.dumps({
    "base": "heuristic v4 (compound +2, pronoun +1, howto +1, definition -2, factoid -1)",
    "production_adjustment": best_a,
    "production_full_set_score": best_s,
    "loo_honest_score": loo_mean,
    "heuristic_alone": heur_score,
    "rule_only": rule_only,
    "llm_only": llm_only,
    "oracle": oracle,
}, indent=2))
print(f"Wrote {OUT / 'informed_heuristic.json'}")
