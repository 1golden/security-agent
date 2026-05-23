#!/usr/bin/env python3
"""RL-4 — Linear CQL-style offline Q-learning.

The trajectory data has 1 terminal reward (final score). So TD targets
collapse to MC returns: Q(s,a) ≈ reward of the trajectory that took
action a in state s, averaged over neighboring states.

Linear Q with CQL penalty:
  L_TD  = Σ (Q(s,a) - return)²
  L_CQL = α * Σ log(Σ_a' exp Q(s,a')) - Q(s,a_data)

At eval time, greedy: a* = argmax_a Q(s,a). Same offline-replay match
as BC, same fixed tie-breaker (mean over ties).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path("/mnt/d/Project/WORK/security-agent/rl/data/replay.jsonl")
OUT = Path("/mnt/d/Project/WORK/security-agent/rl/data/cql_results.json")

FEATURE_KEYS = [
    "q_word_count", "q_short_factoid", "q_definition", "q_howto",
    "q_compound", "q_has_mitre_id", "q_has_cve", "q_has_owasp",
    "q_kw_mitigation", "q_kw_describe", "q_kw_session", "q_kw_logging",
    "retry_round", "n_docs_so_far", "coverage_so_far",
    "missing_count", "has_retrieved",
]


def featurize(state):
    return np.array([float(state.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)


def load_rows():
    return [json.loads(l) for l in open(DATA) if l.strip()]


def fit_cql_q(X, a_idx, returns, n_actions, alpha_cql=0.1, lam=0.3,
              lr=0.05, iters=200, grad_clip=5.0):
    """Linear Q(s,a) = W[a] · [1, x]. CQL-penalized MC regression.

    Fully vectorized + numerically stable (clip td_err, clip grad norm,
    lower lr+iters). Earlier version at lr=0.1, iters=500 diverged on
    full-data refit — see rl/export_cql_weights.py.
    """
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])              # (n, d+1)
    W = np.zeros((n_actions, d + 1))
    a_onehot = np.zeros((n, n_actions), dtype=np.float32)
    a_onehot[np.arange(n), a_idx] = 1.0
    for _ in range(iters):
        Q_all = Xb @ W.T
        Q_taken = (Q_all * a_onehot).sum(axis=1)
        td_err = np.clip(Q_taken - returns, -grad_clip, grad_clip)
        td_grad_for_W = a_onehot * td_err[:, None]
        Qm = Q_all - Q_all.max(axis=1, keepdims=True)
        p = np.exp(Qm)
        p /= p.sum(axis=1, keepdims=True)
        cql_grad_for_W = (p - a_onehot)
        grad_W = (td_grad_for_W + alpha_cql * cql_grad_for_W).T @ Xb / n
        gn = np.linalg.norm(grad_W)
        if gn > grad_clip:
            grad_W *= grad_clip / gn
        W -= lr * grad_W
        W[:, 1:] *= (1.0 - lr * lam / n)
    return W


def predict_action(W, x):
    xb = np.hstack([[1.0], x])
    return int(np.argmax(W @ xb))


def main():
    rows = load_rows()
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
    action_vocab = {
        stg: sorted({r["action"] for r in by_stage[stg]})
        for stg in ("pre", "post", "write")
    }
    label_to_idx = {
        stg: {a: i for i, a in enumerate(action_vocab[stg])}
        for stg in action_vocab
    }

    rows_by_q = defaultdict(list)
    for r in rows:
        rows_by_q[r["q_id"]].append(r)
    q_ids = sorted(rows_by_q)

    q_features_dict = {}
    for q_id in q_ids:
        for r in rows_by_q[q_id]:
            if r["policy"] == "rule":
                q_features_dict[q_id] = {k: r["state"][k] for k in FEATURE_KEYS if k.startswith("q_")}
                break

    # use the same offline-replay match as train_bc.py
    def replay_match(q_id, pred):
        pre_pred, post_pred, write_pred = pred
        by_policy = defaultdict(list)
        for r in rows_by_q[q_id]:
            by_policy[r["policy"]].append(r)
        scored = []
        for pol, rows_ in by_policy.items():
            first = {}
            for r in sorted(rows_, key=lambda r: r["step_idx"]):
                first.setdefault(r["stage"], r["action"])
            s = (int(first.get("pre") == pre_pred)
                 + int(first.get("post") == post_pred)
                 + int(first.get("write") == write_pred))
            scored.append((s, pol, float(rows_[0]["reward"])))
        bs = max(s for s, _, _ in scored)
        tied = [(p, r) for s, p, r in scored if s == bs]
        return ("+".join(sorted(p for p, _ in tied)),
                sum(r for _, r in tied) / len(tied))

    results = {}
    for alpha_cql in (0.0, 0.1, 0.5):
        print(f"\n=== CQL α={alpha_cql} ===")
        loo_scores = []
        loo_arms = Counter()
        for held_q in q_ids:
            Ws = {}
            for stg in ("pre", "post", "write"):
                stage_rows = [r for r in by_stage[stg] if r["q_id"] != held_q]
                X = np.array([featurize(r["state"]) for r in stage_rows], dtype=np.float32)
                a_idx = np.array([label_to_idx[stg][r["action"]] for r in stage_rows], dtype=np.int64)
                returns = np.array([r["reward"] for r in stage_rows], dtype=np.float32)
                Ws[stg] = fit_cql_q(X, a_idx, returns, len(action_vocab[stg]),
                                    alpha_cql=alpha_cql)
            # predict for held_q
            qf = q_features_dict[held_q]
            s_pre = dict(qf); s_pre.update({"retry_round": 0, "n_docs_so_far": 0,
                                            "coverage_so_far": 0, "missing_count": 0, "has_retrieved": 0})
            s_post = dict(s_pre); s_post.update({"n_docs_so_far": 4, "coverage_so_far": 0.45,
                                                  "missing_count": 1, "has_retrieved": 1})
            s_write = dict(s_post); s_write.update({"retry_round": 1})
            pre_a = action_vocab["pre"][predict_action(Ws["pre"], featurize(s_pre))]
            post_a = action_vocab["post"][predict_action(Ws["post"], featurize(s_post))]
            write_a = action_vocab["write"][predict_action(Ws["write"], featurize(s_write))]
            arm, reward = replay_match(held_q, (pre_a, post_a, write_a))
            loo_scores.append(reward)
            loo_arms[arm] += 1
        mean = float(np.mean(loo_scores))
        print(f"  LOO mean: {mean:.4f}  (n={len(loo_scores)})")
        print(f"  arms: {dict(loo_arms)}")
        results[f"alpha_{alpha_cql}"] = {
            "loo_mean": mean, "n": len(loo_scores),
            "arms": dict(loo_arms),
        }

    results["baselines"] = {
        "rule_only": 0.6578,
        "router_v4_heuristic": 0.6733,
        "bc_vanilla": 0.6689,
        "oracle_max_per_q": 0.8366,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
