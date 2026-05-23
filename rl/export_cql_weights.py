#!/usr/bin/env python3
"""Export the production CQL weights (trained on ALL 50 Qs, no holdout).

LOO-CV in train_cql.py was for honest evaluation. For deployment we
re-fit on the full dataset and ship those weights, since the production
router will see new Qs not in the v4 set.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/d/Project/WORK/security-agent")
DATA = ROOT / "rl" / "data" / "replay.jsonl"
OUT = ROOT / "rl" / "data" / "cql_production_weights.json"

FEATURE_KEYS = [
    "q_word_count", "q_short_factoid", "q_definition", "q_howto",
    "q_compound", "q_has_mitre_id", "q_has_cve", "q_has_owasp",
    "q_kw_mitigation", "q_kw_describe", "q_kw_session", "q_kw_logging",
    "retry_round", "n_docs_so_far", "coverage_so_far",
    "missing_count", "has_retrieved",
]


def featurize(state):
    return np.array([float(state.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)


def fit_cql_q(X, a_idx, returns, n_actions, alpha_cql=0.0, lam=0.3,
              lr=0.05, iters=200, grad_clip=5.0):
    """Vectorized linear CQL. Same as train_cql.py's version with lower
    lr + iters + grad clipping for full-data stability (LOO didn't need
    these because removing 1 Q kept the dataset small enough that
    gradients didn't blow up; full 50 Qs at lr=0.1 diverges at iter ~120).
    """
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    W = np.zeros((n_actions, d + 1))
    a_onehot = np.zeros((n, n_actions), dtype=np.float32)
    a_onehot[np.arange(n), a_idx] = 1.0
    for _ in range(iters):
        Q_all = Xb @ W.T
        Q_taken = (Q_all * a_onehot).sum(axis=1)
        td_err = np.clip(Q_taken - returns, -grad_clip, grad_clip)
        td_grad = a_onehot * td_err[:, None]
        Qm = Q_all - Q_all.max(axis=1, keepdims=True)
        p = np.exp(Qm)
        p /= p.sum(axis=1, keepdims=True)
        cql_grad = (p - a_onehot)
        grad = (td_grad + alpha_cql * cql_grad).T @ Xb / n
        gn = np.linalg.norm(grad)
        if gn > grad_clip:
            grad *= grad_clip / gn
        W -= lr * grad
        W[:, 1:] *= (1.0 - lr * lam / n)
    return W


def main():
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)

    action_vocab = {
        stg: sorted({r["action"] for r in by_stage[stg]})
        for stg in ("pre", "post", "write")
    }
    label_to_idx = {stg: {a: i for i, a in enumerate(action_vocab[stg])}
                    for stg in action_vocab}

    weights = {}
    for stg in ("pre", "post", "write"):
        rs = by_stage[stg]
        X = np.array([featurize(r["state"]) for r in rs], dtype=np.float32)
        a_idx = np.array([label_to_idx[stg][r["action"]] for r in rs], dtype=np.int64)
        returns = np.array([r["reward"] for r in rs], dtype=np.float32)
        W = fit_cql_q(X, a_idx, returns, len(action_vocab[stg]), alpha_cql=0.0)
        weights[stg] = {
            "W": W.tolist(),
            "actions": action_vocab[stg],
            "feature_keys": FEATURE_KEYS,
        }
        print(f"{stg}: W shape {W.shape}, actions {action_vocab[stg]}")

    OUT.write_text(json.dumps(weights, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
