# V6 Multi-trial Calibration — Noise Floor + CI Tightening

**Triggered by:** v6 report recommended multi-trial validation to confirm the +0.083 v5 lift was robust. Ran 2 additional Rule-on-v5 50-Q trials.

## Headline (revised)

| Method | Single-trial | 3-trial mean | 3-trial 95% CI |
|---|---|---|---|
| v4 Rule (v4 index, 1 trial) | 0.6578 | – | (unknown) |
| **v5 Rule (v5 index, 3 trials)** | – | **0.7273** | **[0.657, 0.797]** |
| v5 CQL (v5 index, 1 trial) | 0.7084 | – | (unknown) |

**The previously-claimed +0.083 v5-over-v4 lift is overconfident.** Best 3-trial point estimate is **+0.069**, with 95% CI [-0.001, +0.139]. Lower bound essentially equals v4 Rule.

## Per-trial breakdown

| Trial | mean | session_prefix |
|---|---|---|
| t1 | 0.7412 | rule-v5 (from v6 report) |
| t2 | 0.7458 | rule-v5-t2 |
| t3 | **0.6948** | rule-v5-t3 |
| **grand mean** | **0.7273** | – |
| trial sd | 0.0282 | – |
| SE of grand mean | 0.0163 | – |

t3's 0.6948 is the outlier — 0.05 below the other two. Not because of any obvious bug; just within DeepSeek's per-Q variance.

## Per-Q variance — 18/50 Qs swing > 0.3 sd

These Qs **flip between 0.1 and 1.0** across trials with the SAME retrieval and SAME prompt. The only sources of randomness are:
1. DeepSeek's nucleus sampling (we use default temperature 0.2)
2. BGE rerank ties (when two chunks have equal cross-encoder scores)
3. LLM-judge's stochasticity (separate DeepSeek call with default temp)

Top variance Qs:

| Q | Category | t1 | t2 | t3 | sd |
|---|---|---|---|---|---|
| 0  | technique-tactic | 0.10 | 1.00 | 1.00 | 0.52 |
| 13 | authentication | 1.00 | 1.00 | 0.10 | 0.52 |
| 21 | csrf | 1.00 | 1.00 | 0.10 | 0.52 |
| 35 | clickjacking | 1.00 | 0.10 | 1.00 | 0.52 |
| 37 | technique-mitigation | 1.00 | 0.10 | 0.10 | 0.52 |
| 49 | technique-mitigation | 0.10 | 1.00 | 1.00 | 0.52 |
| 4  | technique-description | 0.97 | 1.00 | 0.10 | 0.51 |
| 25 | session | 0.10 | 0.10 | 0.97 | 0.51 |
| 1  | technique-tactic | 1.00 | 0.15 | 1.00 | 0.49 |
| 31 | logging | 0.91 | 0.89 | 0.10 | 0.46 |

The judge appears **bimodal** — most non-1.0 scores cluster at 0.10 (refusal floor). Almost no answers land at e.g. 0.5 or 0.7. So a single LLM token swap can flip a score from 1.0 to 0.1.

## Per-category mean across all 150 samples (3 × 50)

| Category | n | mean |
|---|---|---|
| input-validation | 3 | 1.000 |
| http-headers | 6 | 0.941 |
| technique-description | 21 | 0.940 |
| container-security | 3 | 0.937 |
| injection | 6 | 0.932 |
| jwt | 6 | 0.841 |
| session | 9 | 0.789 |
| xss | 6 | 0.775 |
| logging | 6 | 0.749 |
| technique-tactic | 12 | 0.748 |
| clickjacking | 3 | 0.700 |
| csrf | 6 | 0.677 |
| cryptography | 9 | 0.639 |
| **technique-mitigation** | **42** | **0.583** ← biggest drag |
| access-control | 3 | 0.500 |
| authentication | 9 | 0.458 |

technique-mitigation (n=42, 28% of all samples) at 0.583 is the main constraint on aggregate. Per the v6 diagnosis, this is partly corpus structure (mitigations live in separate STIX objects unreachable by `_lookup_by_external_ids("Txxxx")`) and partly judge calibration.

## What this means for all prior reports

**All single-trial benchmark numbers (v3, v4, v5, v6) carry an unstated ±0.05 noise.** Comparisons between single-trial means with deltas < 0.10 are statistically meaningless.

Re-interpreting prior claims:

| Original claim | Honest reading |
|---|---|
| v4 Rule 0.658 → Heuristic Router 0.673 (+0.015) | Within noise. Heuristic might or might not help. |
| v4 Heuristic Router 0.673 → Oracle 0.837 (+0.16) | Real gap, > noise. Oracle ceiling claim holds. |
| Phase 1 CQL +0.019 LOO over heuristic | Within noise; not a real lift on the LOO methodology either. |
| v5 Rule 0.741 vs v4 Rule 0.658 (+0.083) | Real but smaller — 3-trial estimate +0.069 [CI -0.001, +0.139] |
| v5 CQL 0.708 vs v5 Rule 0.741 (-0.033) | Within noise. CQL might not actually be worse. Would need 3+ CQL trials to verify. |
| v4 Rule mitigation 0.778 vs v5 Rule mitigation 0.583 | Bigger than noise, real regression — but per v6 diagnosis, judge-calibration artifact on legitimate "no info" refusals |

## How much more compute to claim things confidently

Target: ±0.02 aggregate 95% CI (3× tighter than current).

- Need SE of grand mean ≤ 0.02 / 1.96 = 0.0102
- Single-trial sd = 0.0282 → trials needed = (0.0282 / 0.0102)² ≈ 7-8 trials per policy
- 4 policies (Rule v4, Rule v5, CQL v5, Heuristic Router v5) × 8 trials × 50 Qs × ~30s/Q = ~16 hours DeepSeek
- Cost: ~$2 at DeepSeek rates

Alternative: **bigger benchmark.** Per-Q variance is what's noisy; aggregate variance shrinks with sqrt(n). At n=150, single-trial sd would be 0.05 × sqrt(50/150) ≈ 0.029 → similar SE per trial. Doesn't help much unless we ALSO add trials.

The combinatorial winner: **150 Qs × 3 trials × 4 policies** = ~12 hours, ~$1.5, gives both broader coverage and tighter CIs.

## Recommendations

1. **Stop publishing single-trial deltas < 0.10.** They're not reliable enough.
2. **Headline v5 finding survives multi-trial scrutiny — barely.** Best estimate is +0.069 over v4 Rule. Worth shipping IF we accept ±0.05 CI; not safe to claim "+0.08 lift" anymore.
3. **CQL's apparent regression vs v5 Rule is NOT confirmed.** -0.033 single-trial is well within noise. If we ever revive CQL, would need multi-trial validation.
4. **Investment priority for next phase**:
   - **Option A**: expand benchmark to 150 Qs AND multi-trial (~$1.5, ~12h) — proper methodology
   - **Option B**: simpler — calibrate judge to be less bimodal (fewer 1.0/0.1 splits), reducing per-Q sd directly
   - **Option C**: accept current noise as the operating regime, ship v5 retrieval, document the uncertainty
5. **The technique-mitigation category drag (0.583, n=42) is the highest-leverage single fix** if/when we revisit retrieval. Building a technique→course-of-action graph at ingest would target it.

## Files

- `experiments/baseline_2026_05_22_v5/bench50_rule_v5{,_t2,_t3}.jsonl` — 3 Rule-on-v5 trials
- `experiments/baseline_2026_05_22_v5/bench50_rule_v5{,_t2,_t3}.log` — pipeline logs
- `scripts/run_rule_v5_multitrial.sh` — driver for trials 2/3 (trial 1 already done)
