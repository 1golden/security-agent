# Baseline report v2 — after 10 bug fixes

Run on 2026-05-18 against DeepSeek (`deepseek-chat`) as both the answer
generator AND the LLM-as-judge AND the new `LLMCoverage` check. Same 8
questions, same `handmade_rag` retriever (real prebuilt index), same
`inmem` memory.

> v1 report (now superseded): see `BASELINE_REPORT.md`. v1's headline
> numbers were inflated by Bug #1 (the judge had no evidence text); v2
> is the honest baseline.

## TL;DR

| Policy | v1 aggregate | v2 aggregate | Δ v1→v2 |
|--------|--------------|--------------|---------|
| Rule  | 0.865 | **0.596** | −0.27 |
| LLM   | 0.689 | **0.431** | −0.26 |
| Rule − LLM gap | +0.176 | **+0.166** | ≈ unchanged |

**Direction holds, magnitude holds, absolute numbers are humbler.** The
17-point gap between Rule and LLM survived honest judging — that's a
real signal. But both scores dropped ~25 points absolute once the judge
could actually verify claims against evidence text. The v1 absolute
numbers were vibes; v2 is real.

## What changed under the hood (10 fixes)

| # | Bug | Fix |
|---|-----|-----|
| 1 | Judge had `text=""` for every chunk → couldn't verify factual claims | `Pipeline._log` for retrieve now embeds truncated doc text in the observation; `score_trajectories._extract_evidence` reconstructs real evidence from that |
| 2 | `compute_coverage` flagged "missing definition" on docs that explained the topic without using the word | Added content-pattern matching to `_aspect_present`; added `LLMCoverage` strategy; pipeline picks via `SA_COVERAGE_KIND` |
| 3 | `DECOMPOSE` was a silent no-op — subqueries never reached the retriever | `Pipeline._retrieve_with_subqueries` runs each subquery and merges by max score |
| 4 | `LLMPrePolicy` rewrote on every retry, looping forever | `_allowed_actions(state)` dynamically excludes rewrite/decompose/choose_provider when those effects already applied; LLM is told which actions are valid |
| 5 | Probe coref picked `"Based"` (first capitalized word) as the entity for `it` | Stop-list of sentence-initial words; prefer acronyms (≥3 caps) and multi-occurrence tokens; emit nothing if no strong candidate |
| 6 | Token budget only counted answer-generation, not policy calls | `LLMBackend.total_tokens_used` accumulator; pipeline samples it after every LLM-touching stage |
| 7 | A wrong `CHOOSE_PROVIDER` zeroed retrieval and burned a retry | Pipeline auto-retries without the filter when filtered retrieval is empty (does NOT consume an attempt) |
| 8 | `FILTER` path appended to `coverage_history` twice, corrupting `coverage_delta()` | Removed the extra append |
| 9 | `max_retries=N` only gave `N−1` retries (off-by-one) | New `Budget.can_retry` semantics: arg is the **target** attempt number; `max_retries=N` now means N total retries |
| 10 | `LLMJudge` evidence cap was 4000 chars — too tight for 8 chunks | Bumped default to 12000 |

All 79 tests pass (8 new bug-regression tests added).

## Why both scores dropped (and why that's good news)

In v1 the judge was scoring on **vibes** because it had no documents to
ground against. Plausible-looking answers got high factual_accuracy
scores even when they didn't match retrieved evidence. In v2:

- `factual_accuracy`: Rule 0.975 → 0.663 (−0.31), LLM 0.738 → 0.413 (−0.32)
- `evidence_grounding`: Rule 0.962 → 0.650 (−0.31), LLM 0.738 → 0.438 (−0.30)
- `completeness`: Rule 0.538 → 0.275, LLM 0.438 → 0.225

The drop is uniform across dimensions and across policies, which is
consistent with "the judge stopped giving free credit" rather than "the
agent got worse". The two policies' RELATIVE ordering is stable.

## Per-question scores — v2

| # | Question | Rule v2 | LLM v2 | Δ |
|---|----------|---------|--------|---|
| 1 | What is OWASP A03 injection? | 0.10 | **0.42** | +0.32 |
| 2 | How do I mitigate broken access control per OWASP A01? | **0.75** | 0.10 | −0.65 |
| 3 | What does NIST 800-53 AC family cover…? | 0.75 | 0.75 | 0 |
| 4 | Which technique does MITRE ATT&CK T1078 describe? | 0.10 | 0.10 | 0 |
| 5 | Is CVE-2024-1234 a SQL injection? | 1.00 | 1.00 | 0 |
| 6 | Explain MITRE ATT&CK T1078 and how to detect it | **0.75** | 0.10 | −0.65 |
| 7 | And how do I prevent it | **1.00** | 0.10 | −0.90 |
| 8 | What is in the CISA KEV catalog…? | 0.32 | **0.88** | +0.56 |

**LLM-policy is high-variance**: it wins big on 2 questions (Q1: +32,
Q8: +56) and loses big on 3 (Q2/Q6: −65 each, Q7: −90). The variance is
the real story — LLM-policy isn't strictly worse, it's *less reliable*.

## Token cost — now actually tracked (Bug #6)

| Policy | total tokens (8 questions) | mean retrieves / question |
|--------|----------------------------|---------------------------|
| Rule v2 | 28,551 | 2.38 |
| LLM v2  | **51,643** (1.81×)         | 2.50 |

LLM uses ~1.8× the tokens for a strictly worse aggregate score on this
benchmark. The token cost was invisible in v1 because Bug #6 wasn't
fixed yet. This is now a clear loss in two dimensions.

## What this means for "should we train RL"

**Still no.** The conclusion from v1 holds, with stronger evidence:

1. Rule beats LLM on aggregate. Training to imitate LLM still has
   nothing useful to imitate.
2. LLM uses ~2× the tokens. Training a smaller-but-faster model that
   imitates Rule would actually *save* tokens — but a learned policy
   approximating a deterministic rule policy doesn't need RL.
3. The variance result is more interesting: LLM wins on some questions
   (Q1, Q8) by a lot. If you could *train a router* between Rule and
   LLM, that'd beat both pure policies — but that's a one-line
   ensemble, not RL.

The real bottleneck is now visible: both policies struggle on questions
where the retriever's top-N docs are tangentially related (Q2: broken
access control retrieves general AuthN docs; Q4: MITRE T1078 retrieves
unrelated chunks). **Improving retrieval will move both policies more
than improving either policy.**

## Concrete next steps (priority order)

1. **Retrieval audit**: for each question, look at the top-8 docs. Are
   they actually relevant? If not, the rerank step needs work
   (BGE-CrossEncoder threshold, hybrid weight tuning), not the policy.
2. **Router ensemble**: at decision time, run both Rule and LLM
   in parallel, pick by a cheap heuristic (or learned selector). Free
   lunch on Q1+Q8 without losing Q2/Q6/Q7.
3. **Question coverage**: 8 questions is too small for a statistically
   reliable comparison. Expand to ~50 questions from SecBench. The Δ
   could flip on a different question distribution.
4. **Only if retrieval is already good AND ensemble doesn't help**:
   *now* consider training. But the training target should be "imitate
   Rule's conservativeness while keeping LLM's gains on Q1/Q8" — a
   distillation/router task, not pure RL.

## Artifacts

```
experiments/baseline_2026_05_17/         # v1 (superseded)
experiments/baseline_2026_05_18_v2/
  baseline_rule_v2.jsonl     # 8 trajectories, RulePolicy + LLMCoverage
  baseline_llm_v2.jsonl      # 8 trajectories, LLMPolicy + LLMCoverage
  scored_rule_v2.jsonl       # judge verdicts (now sees real evidence)
  scored_llm_v2.jsonl
```

Reproduce:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com
export SA_LLM_BACKEND=openai SA_LLM_MODEL=deepseek-chat
export SA_RETRIEVER_BACKEND=handmade_rag
export SA_HANDMADE_RAG_ROOT=/path/to/all-in-rag/code/handmade_rag
export SA_MEMORY_BACKEND=inmem
export SA_COVERAGE_KIND=llm          # NEW — Bug #2 fix
export SA_LLM_MAX_TOKENS=600 SA_MAX_RETRIES=2

python scripts/collect_trajectories.py --policy rule --out runs/baseline_rule_v2.jsonl
python scripts/collect_trajectories.py --policy llm  --out runs/baseline_llm_v2.jsonl
python scripts/score_trajectories.py runs/baseline_rule_v2.jsonl --judge llm --out runs/scored_rule_v2.jsonl
python scripts/score_trajectories.py runs/baseline_llm_v2.jsonl  --judge llm --out runs/scored_llm_v2.jsonl
```
