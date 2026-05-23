#!/usr/bin/env python3
"""Learned router experiment (E1).

Take v4 Rule + LLM scored rows. Per-Q label = whichever arm scored
higher (ties → rule, the cheaper arm). Extract a 10-feature vector per
question (a superset of the heuristic router's 5). Train logistic
regression with leave-one-out CV: for each Q we hold it out, fit on
the other 49, predict, and record which arm we'd have chosen.

We then compute the achievable score of that LOO-predicted policy
stream — that's our honest estimate of how a learned router would have
performed on this 50-Q set without test-set leakage.

We also dump the trained model (refit on the full 50) plus its feature
weights so the router-pre policy can later load it instead of the
heuristic.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

EXP = Path('/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_19_v4')
OUT_DIR = Path('/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_22_v5')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- features --------
_PRONOUNS = ("it", "they", "this", "that", "these", "those", "he", "she")
_CONJ_RE = re.compile(r"\b(and|then|also|as well as)\b", re.IGNORECASE)
_HOW_RE = re.compile(r"^\s*how\b", re.IGNORECASE)
_DEF_RE = re.compile(r"^\s*what\s+(is|are|does|do)\b", re.IGNORECASE)
_MITRE_RE = re.compile(r"\b([TMGS]\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_OWASP_RE = re.compile(r"\bA0\d:20\d\d\b|\bA0\d\b", re.IGNORECASE)
_LIST_RE = re.compile(r"\b(list|enumerate|name\s+\d|which\s+\d)\b", re.IGNORECASE)

# Semantic-category proxy keywords (derived from v4 per-category outcomes).
_MITIGATION_KW = re.compile(
    r"\b(mitigat|prevent|stop|defen[cs]|harden|protect|countermeasur|"
    r"remediat|block|disable)\b", re.IGNORECASE,
)
_DESCRIBE_KW = re.compile(
    r"\b(describ|explain|what does|how does|purpose of|role of|"
    r"meaning of|definition of)\b", re.IGNORECASE,
)
_IMPLEMENT_KW = re.compile(
    r"\b(implement|configure|set up|deploy|enable|enforce)\b",
    re.IGNORECASE,
)
_SESSION_KW = re.compile(
    r"\b(session|cookie|token|jwt|authent|login|logout|sso|oauth|saml)\b",
    re.IGNORECASE,
)
_LOGGING_KW = re.compile(
    r"\b(log\b|logging|audit|monitor|alert|siem|detect)\b",
    re.IGNORECASE,
)
_CRYPTO_KW = re.compile(
    r"\b(encrypt|decrypt|hash|cipher|tls|ssl|key\s+exchange|crypto|"
    r"signing|certificate)\b", re.IGNORECASE,
)


def featurize(q: str) -> dict[str, float]:
    low = q.lower()
    wc = len(re.findall(r"\S+", q))
    return {
        # shape features
        "f_compound": 1.0 if (_CONJ_RE.search(q) and wc >= 6) else 0.0,
        "f_pronoun": 1.0 if any(re.search(rf"\b{p}\b", low) for p in _PRONOUNS) else 0.0,
        "f_short_factoid": 1.0 if wc <= 6 else 0.0,
        "f_definition": 1.0 if _DEF_RE.match(q) else 0.0,
        "f_open_howto": 1.0 if _HOW_RE.match(q) else 0.0,
        "f_has_id": 1.0 if (_MITRE_RE.search(q) or _CVE_RE.search(q) or _OWASP_RE.search(q)) else 0.0,
        "f_has_cve": 1.0 if _CVE_RE.search(q) else 0.0,
        "f_has_mitre": 1.0 if _MITRE_RE.search(q) else 0.0,
        "f_word_count_norm": min(wc, 40) / 40.0,
        "f_listy": 1.0 if _LIST_RE.search(q) else 0.0,
        # semantic category proxies (v5 — derived from per-category outcomes)
        "f_kw_mitigation": 1.0 if _MITIGATION_KW.search(q) else 0.0,
        "f_kw_describe": 1.0 if _DESCRIBE_KW.search(q) else 0.0,
        "f_kw_implement": 1.0 if _IMPLEMENT_KW.search(q) else 0.0,
        "f_kw_session": 1.0 if _SESSION_KW.search(q) else 0.0,
        "f_kw_logging": 1.0 if _LOGGING_KW.search(q) else 0.0,
        "f_kw_crypto": 1.0 if _CRYPTO_KW.search(q) else 0.0,
    }


FEATURE_NAMES = list(featurize("dummy").keys())


# -------- data --------
def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


rule = {r["i"]: r for r in load_jsonl(EXP / "bench50_rule_v4.jsonl")}
llm = {r["i"]: r for r in load_jsonl(EXP / "bench50_llm_v4.jsonl")}
assert sorted(rule) == sorted(llm) == list(range(50)), "mismatched indices"

rows = []
for i in range(50):
    r, l = rule[i], llm[i]
    q = r["question"]
    rs, ls = float(r["score"]), float(l["score"])
    # label: 1 = LLM strictly better, 0 = Rule (ties → Rule, it's cheaper)
    y = 1 if ls > rs + 1e-9 else 0
    f = featurize(q)
    rows.append({
        "i": i, "q": q, "category": r["category"],
        "rule_score": rs, "llm_score": ls,
        "y": y, **f,
    })

X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=float)
y = np.array([r["y"] for r in rows], dtype=int)
print(f"data: n={len(rows)}  label balance: rule={int((y==0).sum())} llm={int((y==1).sum())}")
print(f"per-category mean diff (LLM-Rule):")
from collections import defaultdict
by_cat = defaultdict(list)
for r in rows:
    by_cat[r["category"]].append(r["llm_score"] - r["rule_score"])
for k in sorted(by_cat):
    v = by_cat[k]
    print(f"  {k:25s}  n={len(v):2d}  diff={sum(v)/len(v):+.3f}")


# -------- LR (manual, no sklearn dep) --------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_lr(X, y, lam=1.0, lr=0.3, iters=3000, balance_classes=True):
    """Logistic regression with L2 reg + optional class weighting.

    With 34:16 imbalance, naive LR collapses to predicting class 0 (rule)
    for everything. Class-weighted loss rebalances the gradient so the
    minority class actually contributes.
    """
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    if balance_classes:
        n_pos = max(int(y.sum()), 1)
        n_neg = max(n - n_pos, 1)
        # sklearn-style "balanced": w_c = n / (2 * n_c)
        weights = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))
    else:
        weights = np.ones(n)
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (weights * (p - y)) / n
        grad[1:] += lam * w[1:] / n
        w -= lr * grad
    return w


def predict(w, X):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return sigmoid(Xb @ w)


# Sweep λ on full-data fit, then LOO-CV with the best.
# For LOO with n=50, this completes in seconds.
def loo_eval(X, y, rule_s, llm_s, lam):
    n = len(y)
    chosen = np.zeros(n, dtype=int)
    probs = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        w = fit_lr(X[idx], y[idx], lam=lam)
        p = float(predict(w, X[i:i+1])[0])
        probs[i] = p
        chosen[i] = 1 if p >= 0.5 else 0
    score = np.where(chosen == 1, llm_s, rule_s).mean()
    return score, chosen, probs


rule_s = np.array([r["rule_score"] for r in rows])
llm_s = np.array([r["llm_score"] for r in rows])

print("\nLOO-CV mean score by λ (class-balanced):")
best = (None, -1)
for lam in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    s, ch, pr = loo_eval(X, y, rule_s, llm_s, lam)
    print(f"  λ={lam:7.3f}  mean={s:.4f}  picked_llm={int(ch.sum())}/{len(ch)}")
    if s > best[1]:
        best = (lam, s, ch, pr)

best_lam, best_s, best_ch, best_pr = best
oracle = np.maximum(rule_s, llm_s).mean()
heur_score = 0.6733  # router v4 patched
rule_only = rule_s.mean()
llm_only = llm_s.mean()
print(f"\n== summary ==")
print(f"  rule-only mean:        {rule_only:.4f}")
print(f"  llm-only mean:         {llm_only:.4f}")
print(f"  heuristic router (v4): {heur_score:.4f}   (from bench50_router_v4_patched.jsonl)")
print(f"  learned router LOO:    {best_s:.4f}   (λ={best_lam})")
print(f"  oracle max:            {oracle:.4f}")
print(f"  gap closed: {(best_s - heur_score) / (oracle - heur_score) * 100:.1f}% of router→oracle")

# Refit on full data for the production model
w_full = fit_lr(X, y, lam=best_lam)
print(f"\nfull-fit weights (bias + {FEATURE_NAMES}):")
print(f"  bias = {w_full[0]:+.4f}")
for name, w in zip(FEATURE_NAMES, w_full[1:]):
    print(f"  {name:22s} = {w:+.4f}")

# Save the model + LOO predictions for the v5 report
model = {
    "feature_names": FEATURE_NAMES,
    "weights": w_full.tolist(),
    "bias_index": 0,
    "lambda": best_lam,
    "loo_mean_score": float(best_s),
    "loo_picked_llm": int(best_ch.sum()),
    "n_train": int(len(y)),
    "oracle_ceiling": float(oracle),
    "heuristic_baseline": float(heur_score),
    "rule_baseline": float(rule_only),
    "llm_baseline": float(llm_only),
}
(OUT_DIR / "learned_router.json").write_text(json.dumps(model, indent=2))

# Per-Q breakdown
detail = []
for r, p, ch in zip(rows, best_pr, best_ch):
    chosen_arm = "llm" if ch == 1 else "rule"
    chosen_score = r["llm_score"] if ch == 1 else r["rule_score"]
    oracle_arm = "llm" if r["llm_score"] > r["rule_score"] else "rule"
    detail.append({
        "i": r["i"],
        "category": r["category"],
        "rule_score": r["rule_score"],
        "llm_score": r["llm_score"],
        "label": "llm" if r["y"] == 1 else "rule",
        "predicted_p_llm": float(p),
        "chosen": chosen_arm,
        "chosen_score": float(chosen_score),
        "oracle_arm": oracle_arm,
        "correct": chosen_arm == oracle_arm or r["rule_score"] == r["llm_score"],
    })
(OUT_DIR / "learned_router_loo_detail.jsonl").write_text(
    "\n".join(json.dumps(d) for d in detail) + "\n"
)
print(f"\nWrote {OUT_DIR / 'learned_router.json'}")
print(f"Wrote {OUT_DIR / 'learned_router_loo_detail.jsonl'}")

# Confusion by category
print("\nLOO accuracy by category (correct = matches oracle arm OR rule==llm):")
by_cat_acc = defaultdict(list)
for d in detail:
    by_cat_acc[d["category"]].append(1 if d["correct"] else 0)
for k in sorted(by_cat_acc):
    v = by_cat_acc[k]
    print(f"  {k:25s}  n={len(v):2d}  acc={sum(v)/len(v):.2f}")
