# RL-1 — Trajectory Data Inventory

**Source**: v4 SAQ benchmark trajectories (only generation with full `.traj.jsonl` files).

| Policy | Trajectories | Total steps | Mean reward |
|---|---|---|---|
| rule | 50 | 800 | 0.658 |
| llm | 50 | 765 | 0.633 |
| router | 46* | 755 | 0.673 |
| **Total** | **146** | **2320** | – |

*Router had 4 hung Qs patched post-hoc; their trajectories are not in the `.traj.jsonl` file. Use the patched `bench50_router_v4_patched.jsonl` for reward labels.

## Pipeline structure per trajectory

```
input_safety → probe → pre → expansion → retrieve → coverage → post
                                                                 ↓ (if gap_retry)
                                                          pre → expansion → retrieve → coverage → post
                                                                                                    ↓ (when answer)
                                                                                              generate → write → output_safety
```

5 of the 10 stages occur 2.18× per trajectory on average (because of `gap_retry` post-actions looping back). Decision-bearing stages:

| Stage | Occurrences | Action vocab (sorted by frequency) |
|---|---|---|
| **pre** | 318 | noop (87%), choose_provider (7%), rewrite (5%), decompose (1%), inherit_filter (<1%) |
| **post** | 318 | gap_retry (62%), answer (32%), narrow (5%), broaden (1%), filter (<1%) |
| **write** | 146 | skip_write (71%), write (29%) |

Other stages are pure observation (input_safety, probe, expansion, retrieve, coverage, generate, output_safety).

## Per-Q oracle policy distribution

For each Q, find the policy whose v4 trajectory scored highest:

| Winning policy | # Qs |
|---|---|
| rule | 29 |
| llm | 14 |
| router | 7 |

Confirms v4 report's "majority class = rule" — any data-driven router trained on this set must beat 29/50 = 0.58 trivially predicting rule. The v4 heuristic router gets 0.6733; oracle ceiling 0.8366.

## RL formulation

**Why step-level beats Q-level (E1)**:
- E1 trained a Q-level policy classifier on n=50 unique Qs → severely underdetermined.
- Step-level trajectories give us **~2000 (state, action) pairs** across 146 trajectories.
- Each stage has its own decision: pre-policy isn't determined by Q-shape alone, it interacts with retrieval observations.

**State features per stage**:
- `pre`: Q shape features (length, has_MITRE_id, has_CVE, definition, etc.) + initial memory count
- `post`: pre features + retrieval observation (`n_docs`, `coverage_score`, `missing_aspects` count from coverage step)
- `write`: post features + final retrieve observation + answer length/token count

**Action labels**: action.type at each decision stage.

**Reward**: per-trajectory final score from LLM-judge (single terminal reward). Reward-to-go at every step = final reward.

**Eval**: LOO-Q cross-validation. For each held-out Q, predict actions at each stage; match predicted action vector to the closest of the 3 v4 trajectories for that Q; use that trajectory's final reward as the achieved score.

## Baselines to beat

| Method | Mean score | Source |
|---|---|---|
| Always Rule | 0.658 | v4 |
| Always LLM | 0.633 | v4 |
| v4 Heuristic Router | 0.673 | v4 |
| v5 LR / rule-enum / informed-heuristic | 0.641–0.658 | E1 (negative) |
| **Per-Q Oracle** | **0.837** | max over 3 v4 policies per Q |

Step-level BC / CQL / DT should target ≥ 0.69 to be considered a "win". Anything <0.673 confirms n=50 ceiling.
