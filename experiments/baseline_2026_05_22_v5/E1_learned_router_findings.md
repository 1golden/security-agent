# E1 — Learned Router (Negative Result)

**Goal**: close some of the v4 Router→Oracle gap (0.673 → 0.837 = 16.4 pt)
by replacing the 5-feature heuristic router with a data-driven selector
trained on v4 Rule/LLM scored outcomes.

## Approaches tried

| Approach | Method | Honest LOO score |
|---|---|---|
| Heuristic (v4) | hand-crafted 5 features, no fit | **0.6733** (production ref) |
| LR (10 shape features) | L2 logistic regression | 0.6578 = rule_only |
| LR (16 shape+kw, class-balanced) | weighted L2 | 0.6412 |
| Single keyword rule | exhaustive search | 0.6573 |
| Informed heuristic (≤2 kw adj) | heuristic + delta | 0.6548 |

Every data-driven method underperforms the heuristic at honest LOO. Most
collapse to rule_only.

## Why

- **Label imbalance**: of 50 Qs, only 16 favor LLM. L2 reg collapses
  weights → majority class. Class-weighted loss over-corrects.
- **Selection variance**: optimal rule on n−1=49 isn't optimal on n=50.
  Rule-enumeration full-set best is 0.6928; LOO honest = 0.6573 — the
  difference is the overfit.
- **No retrieval signal**: per-Q P@5 (the single most predictive
  feature) is only available for 8 Qs (audit set), not 50.

## Production decision

**Keep the heuristic router unchanged** for v5. Wiring `learned_router.json`
into the runtime is not worth it; the cheapest LR pick is strictly
worse than the heuristic on held-out data.

Two strictly-better paths exist but are out of scope here:

1. **Larger benchmark**. Re-run with 150 Qs; that's the minimum where
   16 features × ~50 minority-class examples becomes statistically
   tractable. Estimate $0.30–0.50 DeepSeek + 4–6h.
2. **Per-Q full-stack routing** (not just pre-policy). The 0.837
   oracle comes from per-Q max(rule_run, llm_run) where each entire
   pipeline ran with one policy. Our Router only swaps the pre-stage;
   post/write stay on rule. Realizing the 0.837 ceiling needs a
   policy-stack-level router. Heuristic on full-stack already drops
   to 0.6553 (worse than pre-only 0.6733 because of post×pre
   interactions), so this isn't free either.

## Artifacts

- `scripts/train_learned_router.py` — LR experiment
- `scripts/enumerate_router_rules.py` — single/OR/veto rule search + honest LOO
- `scripts/tune_informed_heuristic.py` — heuristic + keyword adjustments
- `learned_router.json` — full-fit model snapshot (NOT loaded by runtime)
- `informed_heuristic.json` — full-fit adjustment (NOT loaded by runtime)
- `router_v5_honest_loo.jsonl` — per-Q LOO trace

## What this says about v4 conclusions

The v4 report's "train a learned router on v4 outcomes (highest leverage,
half a day)" recommendation is **wrong**. The honest LOO experiments
above show the heuristic IS the local optimum at this dataset size.
The actual highest-leverage move is closing the retrieval gaps (E3/E4).
