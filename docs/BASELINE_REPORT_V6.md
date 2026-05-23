# Baseline Report v6 — Real-Pipeline Attribution + CQL Retrospective

**Date:** 2026-05-23
**Predecessors:** [v5](BASELINE_REPORT_V5.md) (retrieval improvements, CQL learned but not deployed), [RL Phase 1](../rl/REPORT.md) (offline CQL +0.019 over heuristic).

## Headline

| Method | mean (50-Q, single trial) | Δ vs v4 Rule |
|---|---|---|
| v4 Rule (v4 index) | 0.6578 | baseline |
| v4 LLM (v4 index) | 0.6327 | -0.025 |
| v4 Heuristic Router | 0.6733 | +0.015 |
| RL Phase 1 CQL (offline LOO, stable) | 0.6921 | +0.034 (predicted) |
| **v5 Rule (v5 index)** | **0.7412** | **+0.083** ⭐ |
| v5 CQL (v5 index, real pipeline) | 0.7084 | +0.051 |

**Production winner: ship v5 retrieval, keep Rule policy.** v5 Rule gives +0.083 over v4 Rule with no policy change. CQL on v5 underperforms Rule by -0.033 due to distribution shift between training data (v4 trajectories) and deployment (v5 retrieval).

## The reversal: RL added negative value

Phase 1 trained CQL on 146 v4 trajectories and predicted (via offline LOO replay-match eval) that it would lift policy from heuristic 0.6733 → **0.6921** (+0.019). Real-pipeline 50-Q eval confirmed the offline number was *close*: real CQL = **0.7084**, +0.016 better than offline predicted.

But that comparison is wrong, because **both numbers compare to the v4 retrieval baseline (0.658)**. The real baseline at deployment is v5 retrieval, which on its own lifts Rule policy to 0.7412.

Net of attribution:
- **E2 + E3 + memory-leak fix (v5 retrieval): +0.083 over v4 Rule.** ← real lift
- **CQL policy on top of v5 retrieval: -0.033 vs Rule on same retrieval.** ← negative contribution

## Where CQL hurt (and where it helped)

Per-Q diff between CQL_v5 and Rule_v5 (|Δ| > 0.05):

| Direction | n | Σ |Δ| | Categories |
|---|---|---|---|
| Helped (CQL > Rule) | 7 | +3.49 | technique-tactic (1), mitigation (3), auth (1), logging (1), crypto (1) |
| Hurt (CQL < Rule) | 10 | −5.22 | mitigation (2), container (1), xss (1), logging (1), csrf (1), tactic (1), access (1), crypto (1), injection (1) |

The 4 biggest CQL failures:

| Q | Category | Rule_v5 | CQL_v5 | Δ |
|---|---|---|---|---|
| 37 | technique-mitigation | 1.00 | 0.10 | -0.90 |
| 32 | container-security | 0.89 | 0.10 | -0.79 |
| 19 | xss | 1.00 | 0.23 | -0.77 |
| 30 | logging | 0.85 | 0.10 | -0.75 |

In all 4 cases: Rule on v5 retrieval *already had a correct answer*. CQL's policy table said "for this category, v4 Rule failed → delegate to LLM here". On v5 Rule succeeds — but CQL still delegated to LLM, which then made a worse answer (often by over-retrying or filtering useful evidence).

The 4 biggest CQL rescues:

| Q | Category | Rule_v5 | CQL_v5 | Δ |
|---|---|---|---|---|
| 0 | technique-tactic | 0.10 | 1.00 | +0.90 |
| 36 | technique-mitigation | 0.10 | 0.75 | +0.65 |
| 47 | technique-mitigation | 0.10 | 0.75 | +0.65 |
| 49 | technique-mitigation | 0.10 | 0.75 | +0.65 |
| 12 | authentication | 0.10 | 0.73 | +0.63 |

In these cases Rule_v5 still fails (E2 tactic fix is non-deterministic, see v5 report's noise discussion). CQL's "delegate to LLM" call rescues with an LLM-generated rewrite.

## Per-category drill-down (n, rule_v4, rule_v5, cql_v5, Δ_retrieval, Δ_CQL)

```
category                    n    rule_v4    rule_v5     cql_v5     r5-r4    cql-r5
access-control              1      0.900      0.500      0.100    -0.400    -0.400
authentication              3      0.662      0.400      0.612    -0.262    +0.212
clickjacking                1      1.000      1.000      1.000     0.000     0.000
container-security          1      0.100      0.885      0.100    +0.785    -0.785
cryptography                3      0.692      0.598      0.582    -0.093    -0.017
csrf                        2      0.537      0.865      0.562    +0.328    -0.302
http-headers                2      1.000      0.930      0.930    -0.070     0.000
injection                   2      0.550      0.955      0.905    +0.405    -0.050
input-validation            1      1.000      1.000      1.000     0.000     0.000
jwt                         2      0.100      0.893      0.893    +0.793     0.000
logging                     2      0.537      0.880      0.537    +0.343    -0.343
session                     3      0.692      0.692      0.700     0.000    +0.008
technique-description       7      0.611      0.986      0.982    +0.375    -0.004
technique-mitigation       14      0.778      0.570      0.627    -0.209    +0.057
technique-tactic            4      0.509      0.713      0.794    +0.204    +0.081
xss                         2      0.550      0.917      0.530    +0.367    -0.387
```

**Where v5 retrieval clearly helped Rule (positive `r5-r4`)**: container-security (+0.79), jwt (+0.79), injection (+0.41), xss (+0.37), technique-description (+0.38), logging (+0.34), csrf (+0.33), technique-tactic (+0.20).

**Where v5 retrieval hurt Rule (negative `r5-r4`)**: access-control (-0.40, n=1), authentication (-0.26, n=3), technique-mitigation (-0.21, n=14). The mitigation regression is the biggest concern — likely E2 tactic fix accidentally hurt mitigation chunk selection in some cases (since both T-IDs trigger the same `_lookup_by_external_ids` path).

## Why CQL transferred poorly

CQL's policy table maps `predicted_action → (rule | llm)` based on v4 trajectory data. The mapping assumed that:

- Stage actions reflect the underlying retrieval quality
- An action that helped in v4 still helps in v5
- The action vocabulary covers what matters

All three assumptions failed:

1. **Retrieval-quality shift**: v5 retrieval reduces "noisy chunk → wrong answer" cases, so the LLM's "rewrite/decompose" actions trained to compensate for bad v4 chunks now compensate for nothing — they just add latency and occasionally re-rank chunks worse.

2. **State features stale**: The `coverage_so_far` and `n_docs_so_far` distributions differ between v4 and v5. CQL trained on v4 expects (say) coverage=0.3 to mean "retry"; on v5 the same coverage value means "this is fine, answer".

3. **Action vocab too narrow**: We already flagged this in Phase 1. The differentiation between policies happens in *rewrite text content*, not in `action.type`. CQL never sees that level.

## Decision tree

| Goal | Action |
|---|---|
| Ship best policy today | **v5 Rule** — 0.7412, no model dependency, deterministic |
| Recover CQL's promise | Retrain CQL on v5 trajectories (cost: 3 policies × 50 Qs × ~20min = 6h DeepSeek + ~$0.5). Only worth it if benchmark expands so we have more LLM-wins examples. |
| Maximize aggregate | Wait for 150-Q benchmark (RL-10), retrain everything fresh, then re-evaluate |
| Validate v5 lift is real | Multi-trial Rule v5 (3× 50-Q) to bound DeepSeek noise. ~50min × 3 = 2.5h. Aggregate noise should be ±0.025 → 3 trials → ±0.014. |

## Methodology lesson

**Always re-baseline before deploying a learned policy.** Offline LOO eval on stale trajectories systematically overestimates lift when the upstream stack improves. The Phase 1 report's "CQL +0.019 over heuristic" was true *for v4 retrieval* — but by the time we deployed, the comparison should have been against the new retrieval baseline, not the old policy baseline.

In RL terms: **the data-collection policy distribution must approximate the deployment distribution**. We violated this by training on v4 and deploying on v5.

## Files added

- `src/security_agent/policies/staged_router.py` — CQLPrePolicy / CQLPostPolicy / CQLWritePolicy
- `src/security_agent/policies/{pre_retrieval,post_retrieval,memory_write}/__init__.py` — `cql` build target
- `scripts/run_benchmark.py` — `--policy cql` choice
- `rl/export_cql_weights.py` — full-data refit with gradient clipping (LOO trainer also stabilized)
- `experiments/baseline_2026_05_22_v5/bench50_cql_v5.jsonl` — 50-Q CQL real-pipeline output
- `experiments/baseline_2026_05_22_v5/bench50_rule_v5.jsonl` — 50-Q Rule on v5 index (attribution baseline)
- `rl/data/cql_production_weights.json` — bounded weights (was 1e+254 before fix)

## Recommended next moves

Pick by appetite:

1. **Ship v5 Rule + stop.** Update production config to use v5 index + Rule policy. +0.083 over v4, no ML dependency. Reproducible.
2. **Multi-trial v5 Rule to confirm.** 3× 50-Q runs of Rule on v5 index to bound noise. ~2.5h DeepSeek. Confirms v5 lift before committing.
3. **Expand benchmark.** 50 → 150 Qs (1 day), then retrain CQL on v5 traj. Higher confidence but ~6h compute.
4. **Targeted CQL fix.** Patch the action→policy mapping so the 4 big regressors (q=37/32/19/30) stay on Rule. ~30min, but feels like hand-tuning around symptoms.
