#!/usr/bin/env python3
"""Rule-enumeration router search.

LR on n=50 with shape+kw features couldn't match the heuristic's 0.673.
This script asks a simpler question: what's the best *single keyword
rule* of the form "pick LLM iff regex R matches the question"?

For each candidate regex (drawn from category-derived keyword groups
and the original heuristic's features), compute the mean LOO-scored
benchmark — except LOO is trivial for a fixed rule (no fitting needed),
so the score IS the LOO score.

Then search 2-rule combinations: "pick LLM iff (R1 matches) AND (R2
does not match)" — i.e. rule + negative-veto. Three patterns:
  pure       : R+
  positive_2 : R1 OR R2
  veto       : R+ AND NOT R-

Best rule is the production router.
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

EXP = Path('/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_19_v4')
OUT = Path('/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_22_v5')
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


rule_rows = load(EXP / 'bench50_rule_v4.jsonl')
llm_rows = load(EXP / 'bench50_llm_v4.jsonl')
qs = [(r['question'], r['score'], l['score'], r['category'])
      for r, l in zip(rule_rows, llm_rows)]
n = len(qs)
rule_only = sum(r for _, r, _, _ in qs) / n
llm_only = sum(l for _, _, l, _ in qs) / n
oracle = sum(max(r, l) for _, r, l, _ in qs) / n
print(f"rule_only={rule_only:.4f}  llm_only={llm_only:.4f}  oracle={oracle:.4f}")
print(f"heuristic (v4 router)={0.6733:.4f}")

CANDIDATES = {
    # category-proxy keywords
    "kw_describe":      r"\b(describ|explain|purpose of|role of|what does|how does)\b",
    "kw_mitigation":    r"\b(mitigat|prevent|defen[cs]|harden|protect|countermeasur|remediat)\b",
    "kw_clickjacking":  r"\b(clickjack|frame busting|x-frame)\b",
    "kw_access":        r"\b(role-based|rbac|access control|principle of least)\b",
    "kw_session":       r"\b(session|cookie attribute|httponly|samesite)\b",
    "kw_logging":       r"\b(log\b|logging|audit|monitor|alert|siem|detect)\b",
    "kw_jwt":           r"\b(jwt|json web token|jws|jwe)\b",
    "kw_container":     r"\b(container|docker|kubernetes|k8s|pod)\b",
    "kw_crypto":        r"\b(encrypt|cipher|tls|ssl|certificate|hash|hmac)\b",
    "kw_xss":           r"\b(xss|cross-?site script)\b",
    # shape-based
    "sh_compound":      r"\b(and|then|also|as well as)\b",
    "sh_definition":    r"^\s*what\s+(is|are|does|do)\b",
    "sh_howto":         r"^\s*how\b",
    "sh_mitre_id":      r"\b([TMGS]\d{4}(?:\.\d{3})?)\b",
    "sh_cve":           r"\bcve-\d{4}-\d{4,7}\b",
    "sh_owasp":         r"\ba0\d(?::20\d\d)?\b",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in CANDIDATES.items()}


def score_with_rule(picks_llm):
    """picks_llm: list[bool] of length n"""
    return sum(l if pick else r for (_, r, l, _), pick in zip(qs, picks_llm)) / n


# -- 1) single positive rule --
results = []
for name, rgx in COMPILED.items():
    picks = [bool(rgx.search(q)) for q, _, _, _ in qs]
    s = score_with_rule(picks)
    n_llm = sum(picks)
    results.append(("pure", name, None, s, n_llm))

# -- 2) two-positive (OR) --
for (a, ra), (b, rb) in combinations(COMPILED.items(), 2):
    picks = [bool(ra.search(q) or rb.search(q)) for q, _, _, _ in qs]
    s = score_with_rule(picks)
    n_llm = sum(picks)
    results.append(("or", a, b, s, n_llm))

# -- 3) positive AND NOT (veto) --
for (a, ra), (b, rb) in [(x, y) for x in COMPILED.items() for y in COMPILED.items() if x != y]:
    picks = [bool(ra.search(q) and not rb.search(q)) for q, _, _, _ in qs]
    s = score_with_rule(picks)
    n_llm = sum(picks)
    results.append(("veto", a, b, s, n_llm))

# -- baseline: always rule, always llm --
results.append(("baseline", "always_rule", None, rule_only, 0))
results.append(("baseline", "always_llm", None, llm_only, n))

# rank
results.sort(key=lambda x: -x[3])
print("\nTop 20 rule combinations:")
print(f"{'kind':10s} {'rule':24s} {'veto/and':22s} {'score':>7s} {'#llm':>5s}")
for kind, a, b, s, nl in results[:20]:
    b_s = b or ""
    star = " *" if s > 0.6733 else ""
    print(f"{kind:10s} {a:24s} {b_s:22s} {s:7.4f} {nl:5d}{star}")

print(f"\nBest: {results[0][:3]} → {results[0][3]:.4f}  vs heuristic 0.6733  vs oracle {oracle:.4f}")

# Save full list
with open(OUT / 'rule_search.jsonl', 'w') as f:
    for kind, a, b, s, nl in results:
        f.write(json.dumps({
            "kind": kind, "rule": a, "veto_or_and": b,
            "mean_score": s, "n_picked_llm": nl,
        }) + "\n")

# ============ HONEST LOO: refit rule choice on n-1 each time ============
# For each held-out i, find the rule maximizing mean score on the other
# 49, then apply that rule to i. The achieved score on i is what counts.
def all_rules():
    rules = []
    for name, rgx in COMPILED.items():
        rules.append(("pure", name, None, rgx, None))
    for (a, ra), (b, rb) in combinations(COMPILED.items(), 2):
        rules.append(("or", a, b, ra, rb))
    for (a, ra), (b, rb) in [(x, y) for x in COMPILED.items()
                              for y in COMPILED.items() if x != y]:
        rules.append(("veto", a, b, ra, rb))
    return rules


RULES = all_rules()


def rule_picks(rule, questions):
    kind, _, _, ra, rb = rule
    if kind == "pure":
        return [bool(ra.search(q)) for q in questions]
    if kind == "or":
        return [bool(ra.search(q) or rb.search(q)) for q in questions]
    if kind == "veto":
        return [bool(ra.search(q) and not rb.search(q)) for q in questions]
    return [False] * len(questions)


print("\n== honest LOO (rule selected on n-1 each fold) ==")
loo_scores = []
loo_rules_chosen = []
all_q_text = [q for q, _, _, _ in qs]
all_rs = [r for _, r, _, _ in qs]
all_ls = [l for _, _, l, _ in qs]
for i in range(n):
    train_q = [all_q_text[j] for j in range(n) if j != i]
    train_rs = [all_rs[j] for j in range(n) if j != i]
    train_ls = [all_ls[j] for j in range(n) if j != i]
    best_rule = None
    best_score = -1
    for rule in RULES:
        picks = rule_picks(rule, train_q)
        s = sum(l if p else r for p, r, l in zip(picks, train_rs, train_ls)) / (n - 1)
        if s > best_score:
            best_score = s
            best_rule = rule
    pick_i = rule_picks(best_rule, [all_q_text[i]])[0]
    achieved_i = all_ls[i] if pick_i else all_rs[i]
    loo_scores.append(achieved_i)
    loo_rules_chosen.append((best_rule[0], best_rule[1], best_rule[2]))

loo_mean = sum(loo_scores) / len(loo_scores)
print(f"  LOO-honest mean = {loo_mean:.4f}  (heuristic {0.6733:.4f}, oracle {oracle:.4f})")
print(f"  unique rules selected across folds: {len(set(loo_rules_chosen))}")
from collections import Counter
top_rules = Counter(loo_rules_chosen).most_common(5)
for rule_key, ct in top_rules:
    print(f"    {ct:2d}x  {rule_key}")

# Save honest LOO detail
honest_detail = []
for i, ((q, rs, ls, cat), achieved, rule_key) in enumerate(zip(qs, loo_scores, loo_rules_chosen)):
    honest_detail.append({
        "i": i, "category": cat, "rule_score": rs, "llm_score": ls,
        "chosen_score": achieved,
        "loo_rule": list(rule_key),
    })
with open(OUT / 'router_v5_honest_loo.jsonl', 'w') as f:
    for d in honest_detail:
        f.write(json.dumps(d) + "\n")
print(f"  wrote {OUT / 'router_v5_honest_loo.jsonl'}")

# Apply BEST GLOBAL rule (training on full set) for the production router
print("\n== production rule (trained on full 50, will be used in v5 pipeline) ==")
best = results[0]
kind, a, b, _, _ = best
ra = COMPILED.get(a) if a in COMPILED else None
rb = COMPILED.get(b) if b in COMPILED else None
if kind == "pure":
    pick_fn = lambda q: bool(ra.search(q))
elif kind == "or":
    pick_fn = lambda q: bool(ra.search(q) or rb.search(q))
elif kind == "veto":
    pick_fn = lambda q: bool(ra.search(q) and not rb.search(q))
else:
    pick_fn = lambda q: kind == "always_llm"

detail = []
for i, (q, rs, ls, cat) in enumerate(qs):
    picked_llm = pick_fn(q)
    arm = "llm" if picked_llm else "rule"
    achieved = ls if picked_llm else rs
    oracle_arm = "llm" if ls > rs else "rule"
    detail.append({
        "i": i, "category": cat, "question": q,
        "rule_score": rs, "llm_score": ls,
        "chosen": arm, "chosen_score": achieved,
        "oracle_arm": oracle_arm, "matches_oracle": arm == oracle_arm or rs == ls,
    })
with open(OUT / 'router_v5_loo_detail.jsonl', 'w') as f:
    for d in detail:
        f.write(json.dumps(d) + "\n")

from collections import defaultdict
by_cat = defaultdict(list)
for d in detail:
    by_cat[d['category']].append(d['chosen_score'])
print("\nv5 per-category:")
for k in sorted(by_cat):
    v = by_cat[k]
    print(f"  {k:25s}  n={len(v):2d}  mean={sum(v)/len(v):.4f}")
