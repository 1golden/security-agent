# Trajectory Replay-based RL — Findings (Phase 1)

**Date:** 2026-05-23
**Predecessor:** [BASELINE_REPORT_V5.md](../docs/BASELINE_REPORT_V5.md). E1 (Q-level learned router) was negative; this work moves to step-level policy learning from existing v4 trajectories.

## TL;DR

Linear CQL trained on per-stage (state, action, terminal-reward) tuples beats the v4 heuristic router by **+0.023** on the same 50-Q SAQ benchmark — the first data-driven router that has cleared the heuristic on this dataset.

| Method | LOO mean | Δ vs heuristic | Action source |
|---|---|---|---|
| Always Rule | 0.6578 | -0.016 | trivial |
| Always LLM | 0.6327 | -0.041 | trivial |
| v4 Heuristic Router | **0.6733** | — | hand-crafted 5-feature |
| E1 LR (Q-level, no traj) | 0.6578 | -0.016 | shape features |
| BC vanilla (step-level) | 0.6689 | -0.004 | imitation, traj data |
| BC reward-weighted | 0.6158 | -0.058 | 1-step REINFORCE-ish |
| **CQL α=0 (step-level)** | **0.6964** | **+0.023** | offline Q regression |
| CQL α=0.1 / α=0.5 | 0.6921 | +0.019 | + conservative penalty |
| Per-Q Oracle ceiling | 0.8366 | +0.163 | max over 3 v4 policies |

CQL closes **14% of the router→oracle gap** with a 50-parameter linear model trained in seconds.

## Why step-level beat Q-level (where E1 failed)

E1 trained a Q-level classifier on 50 unique question-feature vectors → severely underdetermined for any non-trivial decision boundary.

Step-level RL widens the training set by ~16× (50 Qs → 782 step-rows) and adds **4 dynamic state features** that vary within a trajectory:

| Feature | Source | Why it matters |
|---|---|---|
| `retry_round` | post-stage gap_retry counter | round-2 decisions face different evidence |
| `n_docs_so_far` | last retrieve obs | low n → need different post-action |
| `coverage_so_far` | last coverage obs | high coverage → "answer" is right call |
| `missing_count` | coverage missing_aspects | high → "gap_retry" or "rewrite" |

These let the model condition on "what the pipeline has seen", not just "what the question looks like". The heuristic router ignores all of this — it picks pre-policy from Q text alone.

## Why CQL beat BC

| Stage | BC behavior | CQL behavior | Difference |
|---|---|---|---|
| pre | predicts `noop` always (87% majority) | predicts `noop` 92% but switches to `decompose`/`rewrite` when retry_round=1 + low coverage | dynamic conditioning |
| post | predicts `gap_retry` if missing_count > 0 else `answer` | predicts `answer` when coverage_so_far > 0.5 else `gap_retry` | smoother threshold |
| write | predicts `write` only when answer length > median | mostly `skip_write` (no good Q-value gradient — terminal reward is confounded) | reverse-causation noise survived in BC |

**BC overfits to action frequency**; CQL fits to action *return*, so a rare-but-effective action like `decompose` (n=4, mean reward 1.0) gets a high Q value and triggers when state matches. BC under-weighs it because it's only 4/318 of the data.

## Why α=0 (no CQL penalty) wins over α>0

CQL's conservative penalty exists to keep policies in-distribution — important when the offline dataset is small relative to the action space, otherwise the learned Q overestimates OOD actions. Here:

- Action space is small (5 / 5 / 2)
- All in-distribution actions ARE seen in the dataset
- There's no "out-of-distribution" action to suppress
- The penalty just adds noise to the otherwise-clean MC regression

α=0.1 → 0.6921, α=0.5 → 0.6921 (tied). The penalty term is dominated by L2 reg, doesn't change the argmax. α=0 (0.6964) ekes out the win.

## Per-Q tie distribution (where CQL helps vs not)

Replay-match uses fair tie-broken eval (mean reward across tied policies), so a single-policy prediction is "decisive" while a multi-policy prediction is "indecisive".

| # ties | Cases | Mean reward | Interpretation |
|---|---|---|---|
| 1 policy (decisive) | 12/50 | 0.74 | CQL extracted a signal |
| 2 policies (partial) | 17/50 | 0.69 | partial signal, hedged |
| 3 policies (all tie) | 21/50 | 0.66 | no actionable signal |

The 12 decisive cases pull the aggregate from 0.66 (no signal) up to 0.6964. This bounds the maximum lift possible from a deterministic linear policy on this dataset.

## Limits exposed

1. **Reverse causation in `write` stage**: `write` action (n=42) has mean reward 0.843 vs `skip_write` (n=104) at 0.583. This is NOT actionable — `write` happens AFTER `generate`, conditional on the answer being good enough to memorize. Picking `write` doesn't *cause* a good answer; a good answer causes the system to pick `write`. CQL learned to (mostly) ignore this stage.

2. **Action vocabulary too narrow**: 87% of pre-actions are `noop`. The 3 v4 policies barely differ in their stage-action vectors; differences are in *rewrite content* (free-text query reformulation), which we don't model. To unlock the remaining 14% router→oracle gap, we'd need to learn over rewritten queries, not just action types.

3. **n=50 unique Qs limits everything**. The 0.6964 → 0.8366 gap won't close without more diverse trajectories. Either: (a) bigger benchmark, (b) trajectory replay with non-trivial action perturbations (counterfactual rollouts), (c) per-Q ensemble (cheap, but needs $$$ for more arms).

## Decision

**Promote CQL α=0 to a production policy candidate** — but only after:

1. Multi-trial validation: with DeepSeek ±0.2 per-Q noise, a +0.023 lift needs 3+ trials of the full 50-Q eval pipeline to claim with confidence. Pre-registering this is the right move before shipping.
2. Real-pipeline eval (not offline replay-match): wire CQL into `RouterPrePolicy.choose()` (or a new `StagedRouterPolicy` covering pre/post/write) and run a fresh 50-Q benchmark. Replay-match is an offline proxy; ground truth is the actual pipeline score.
3. Larger benchmark before re-evaluation. Same recommendation as v5 — n=50 isn't enough to ship a +0.023 lift safely.

Net: the **infrastructure is now in place** (`replay_dataset.py`, `train_bc.py`, `train_cql.py`, LOO-Q eval) so that when (3) lands, we re-train in seconds and re-evaluate cheaply. That's the real win of this Phase 1 — a working RL stack on a real agent trajectory dataset.

## Files

- `rl/inventory.md` — trajectory data inventory
- `rl/replay_dataset.py` — extract (state, action, reward) from .traj.jsonl
- `rl/data/replay.jsonl` — 782 step-level rows
- `rl/data/dataset_stats.txt` — per-stage action × reward stats
- `rl/train_bc.py` — vanilla + reward-weighted behavior cloning
- `rl/train_cql.py` — linear CQL (vectorized)
- `rl/data/bc_results.json`, `rl/data/cql_results.json` — LOO-CV metrics
