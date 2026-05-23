#!/usr/bin/env python3
"""RL-3 — Behavior cloning baseline + offline replay eval.

Train one classifier per decision stage (pre / post / write) that predicts
the action.type taken in the trajectory. Two training variants:

  1. Vanilla BC — supervised on ALL (state, action) pairs equally.
  2. Reward-weighted BC — weight loss by (reward - baseline) so rare-but-
     high-reward actions get amplified. Baseline = mean reward at this
     stage. Effectively a 1-step REINFORCE.

**Evaluation (offline replay match)**: for each held-out Q, we predict an
action at each stage. We then find which of the 3 v4 trajectories for
that Q has the closest action vector to our predictions, and use that
trajectory's reward as our achieved score. This is "what policy would
have run if we listened to the BC model".

  match metric: count of (stage, action) matches against each candidate
  trajectory; ties broken by reward (favor the higher-reward arm).

This is the same offline-eval trick from off-policy bandits — we can
only score what we have observations for. Caveat: if BC predicts an
action that NO v4 trajectory took on this Q, we fall back to the
closest action match per Hamming distance.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path("/mnt/d/Project/WORK/security-agent/rl/data/replay.jsonl")
OUT = Path("/mnt/d/Project/WORK/security-agent/rl/data/bc_results.json")

# Feature ordering — must be stable across train and test.
FEATURE_KEYS = [
    "q_word_count", "q_short_factoid", "q_definition", "q_howto",
    "q_compound", "q_has_mitre_id", "q_has_cve", "q_has_owasp",
    "q_kw_mitigation", "q_kw_describe", "q_kw_session", "q_kw_logging",
    "retry_round", "n_docs_so_far", "coverage_so_far",
    "missing_count", "has_retrieved",
]


def featurize(state: dict) -> np.ndarray:
    return np.array([float(state.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)


def load_rows() -> list[dict]:
    return [json.loads(l) for l in open(DATA) if l.strip()]


# ---------------------------------------------------------------------------
# Multinomial logistic regression (manual; sklearn-free)
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_mlr(X, y, n_classes, lam=0.3, lr=0.3, iters=2000, weights=None):
    """Multinomial LR with L2, gradient descent. Tiny problem."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    W = np.zeros((d + 1, n_classes))
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0
    w = weights if weights is not None else np.ones(n)
    w = w / w.sum() * n  # normalize so lr scaling stays sane
    for _ in range(iters):
        P = softmax(Xb @ W)
        diff = (P - Y) * w[:, None]
        grad = Xb.T @ diff / n
        grad[1:] += lam * W[1:] / n
        W -= lr * grad
    return W


def predict_mlr(W, X):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return softmax(Xb @ W)


# ---------------------------------------------------------------------------
# Per-stage training + LOO-Q eval
# ---------------------------------------------------------------------------
def train_stage(stage_rows, label_to_idx, weight_mode="none"):
    X = np.array([featurize(r["state"]) for r in stage_rows], dtype=np.float32)
    y = np.array([label_to_idx[r["action"]] for r in stage_rows], dtype=np.int64)
    if weight_mode == "reward":
        # weights = max(reward - 0.5, 0.1) — upweight high-reward examples,
        # but never zero out anything (smooth floor at 0.1).
        rewards = np.array([r["reward"] for r in stage_rows], dtype=np.float32)
        weights = np.maximum(rewards - 0.5, 0.1)
    else:
        weights = None
    W = fit_mlr(X, y, len(label_to_idx), weights=weights)
    return W


def predict_action_for_q(W_pre, W_post, W_write, q_features_dict, action_vocab):
    """Greedy: at each stage take argmax. For non-pre stages we don't have
    'true' downstream observations until we replay — so we use *expected*
    values seen at training time as proxies.

    For offline replay matching, this works: we get a 3-tuple of predicted
    actions per Q, then match.
    """
    # pre — use Q features only (no retry, no retrieval yet)
    s_pre = dict(q_features_dict)
    s_pre.update({"retry_round": 0.0, "n_docs_so_far": 0.0,
                  "coverage_so_far": 0.0, "missing_count": 0.0,
                  "has_retrieved": 0.0})
    pre_idx = int(np.argmax(predict_mlr(W_pre, featurize(s_pre)[None, :])))
    pre_action = action_vocab["pre"][pre_idx]

    # post — we don't know the actual retrieval result; use median training values
    s_post = dict(s_pre)
    s_post.update({"n_docs_so_far": 4.0, "coverage_so_far": 0.45,
                   "missing_count": 1.0, "has_retrieved": 1.0})
    post_idx = int(np.argmax(predict_mlr(W_post, featurize(s_post)[None, :])))
    post_action = action_vocab["post"][post_idx]

    # write
    s_write = dict(s_post)
    s_write.update({"retry_round": 1.0})
    write_idx = int(np.argmax(predict_mlr(W_write, featurize(s_write)[None, :])))
    write_action = action_vocab["write"][write_idx]

    return pre_action, post_action, write_action


def replay_match_score(q_id, predicted_actions, rows_by_q):
    """Given (pre, post, write) prediction, find which v4 trajectory for
    this Q matches best and return its reward.

    Matching: for each candidate policy, summarize the trajectory by its
    FIRST-occurrence action per stage (so retry-induced repeats don't
    count). Compare to predicted 3-tuple; score = # matches.

    Tie-break: when multiple policies tie on action-match score, return
    the MEAN reward across the tied set. This avoids the oracle-leak
    bug where "pick the higher-reward arm on tie" silently picks the
    winning arm whenever predictions are uninformative.
    """
    pre_pred, post_pred, write_pred = predicted_actions
    by_policy = defaultdict(list)
    for r in rows_by_q[q_id]:
        by_policy[r["policy"]].append(r)
    if not by_policy:
        return None, None

    scored: list[tuple[int, str, float]] = []
    for pol, rows in by_policy.items():
        first_per_stage = {}
        for r in sorted(rows, key=lambda r: r["step_idx"]):
            first_per_stage.setdefault(r["stage"], r["action"])
        s = 0
        s += int(first_per_stage.get("pre") == pre_pred)
        s += int(first_per_stage.get("post") == post_pred)
        s += int(first_per_stage.get("write") == write_pred)
        scored.append((s, pol, float(rows[0]["reward"])))

    best_score = max(s for s, _, _ in scored)
    tied = [(p, r) for s, p, r in scored if s == best_score]
    # tie-break: mean reward across all tied policies (no oracle leak).
    # If only one policy ties (clean signal), this == its reward.
    mean_reward = sum(r for _, r in tied) / len(tied)
    pol_label = "+".join(sorted(p for p, _ in tied))
    return pol_label, mean_reward


def main():
    rows = load_rows()
    # build stage-specific datasets
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)

    # action vocabs (sorted for determinism)
    action_vocab = {
        stg: sorted({r["action"] for r in by_stage[stg]})
        for stg in ("pre", "post", "write")
    }
    print(f"Action vocabs: {action_vocab}")
    label_to_idx = {
        stg: {a: i for i, a in enumerate(action_vocab[stg])}
        for stg in action_vocab
    }

    # group rows by q_id for replay-match eval
    rows_by_q = defaultdict(list)
    for r in rows:
        rows_by_q[r["q_id"]].append(r)
    q_ids = sorted(rows_by_q)

    # Pre-compute Q-only features (q_id → feature dict) by pulling from first
    # rule-policy row per Q (Q features are stable across policies).
    q_features_dict = {}
    for q_id in q_ids:
        for r in rows_by_q[q_id]:
            if r["policy"] == "rule":
                # extract Q-only keys
                qf = {k: r["state"][k] for k in FEATURE_KEYS if k.startswith("q_")}
                q_features_dict[q_id] = qf
                break

    results = {}

    for weight_mode in ("none", "reward"):
        print(f"\n========== BC weight_mode={weight_mode} ==========")
        loo_scores = []
        loo_predicted_policy = Counter()
        for held_q in q_ids:
            train_idx_by_stage = {
                stg: [r for r in by_stage[stg] if r["q_id"] != held_q]
                for stg in ("pre", "post", "write")
            }
            W_pre = train_stage(train_idx_by_stage["pre"],
                                label_to_idx["pre"], weight_mode)
            W_post = train_stage(train_idx_by_stage["post"],
                                 label_to_idx["post"], weight_mode)
            W_write = train_stage(train_idx_by_stage["write"],
                                  label_to_idx["write"], weight_mode)
            pred = predict_action_for_q(W_pre, W_post, W_write,
                                        q_features_dict[held_q], action_vocab)
            best_pol, achieved = replay_match_score(held_q, pred, rows_by_q)
            if achieved is None:
                continue
            loo_scores.append(achieved)
            loo_predicted_policy[best_pol] += 1
            if held_q < 5:
                print(f"  q={held_q}  pred_actions={pred}  matched={best_pol}  reward={achieved:.3f}")

        mean = float(np.mean(loo_scores))
        print(f"  ---")
        print(f"  LOO mean reward: {mean:.4f}  (n={len(loo_scores)})")
        print(f"  Matched-policy distribution: {dict(loo_predicted_policy)}")
        results[weight_mode] = {
            "loo_mean": mean,
            "n": len(loo_scores),
            "matched_policy_counts": dict(loo_predicted_policy),
        }

    # baselines for comparison
    results["baselines"] = {
        "rule_only": 0.6578,
        "llm_only": 0.6327,
        "router_v4_heuristic": 0.6733,
        "oracle_max_per_q": 0.8366,
    }

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
