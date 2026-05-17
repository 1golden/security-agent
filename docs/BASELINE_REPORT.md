# Baseline report — Rule vs LLM policy

Run on 2026-05-17 against DeepSeek (`deepseek-chat`) as both the answer
generator and the LLM-as-judge, with `handmade_rag` retriever (real
prebuilt `security_index`) and `inmem` memory.

## Setup

- Questions: 8 builtin questions (see `scripts/collect_trajectories.py`)
- Backends: handmade_rag retriever + inmem memory + DeepSeek for both
  answer generation and policy decisions
- Budget: `max_retries=2`, `max_tokens=600`
- Judge: LLMJudge over DeepSeek (same model — known self-eval bias risk;
  relative ordering still valid)

## Headline numbers

| Policy | Aggregate | factual | grounding | completeness | safety |
|--------|-----------|---------|-----------|--------------|--------|
| **Rule**  | **0.865** | **0.975** | **0.962** | 0.538 | 1.0 |
| **LLM**   | 0.689     | 0.738     | 0.738     | 0.438 | 1.0 |
| Δ      | **−0.176**| −0.237    | −0.225    | −0.100       |  0    |

**Rule policy decisively beats LLM policy by 17.6 points.** Driven by
factual accuracy and evidence grounding — the LLM is somehow producing
answers that are less faithful to the retrieved evidence.

## Per-question scores

| # | Question | Rule | LLM | Δ |
|---|----------|------|-----|---|
| 1 | What is OWASP A03 injection? | 1.00 | **0.10** | −0.90 |
| 2 | How do I mitigate broken access control per OWASP A01? | 0.75 | 0.75 | 0 |
| 3 | What does NIST 800-53 AC family cover…? | 0.75 | 0.75 | 0 |
| 4 | Which technique does MITRE ATT&CK T1078 describe? | 1.00 | 1.00 | 0 |
| 5 | Is CVE-2024-1234 a SQL injection? | 1.00 | 1.00 | 0 |
| 6 | Explain MITRE ATT&CK T1078 and how to detect it | 1.00 | **0.10** | −0.90 |
| 7 | And how do I prevent it | 1.00 | 0.84 | −0.16 |
| 8 | What is in the CISA KEV catalog…? | 0.67 | 0.97 | +0.30 |

LLM wins one (Q8), ties three, loses four — including two catastrophic
(−0.9). Median answer length: LLM is ~2× longer than Rule (562 vs 330)
but length didn't translate to score.

## Root cause for the LLM losses

Trace inspection on Q1 ("What is OWASP A03 injection?"):

```
pre      rewrite       payload={"query": "OWASP A03 injection vulnerability definition and..."}
                         | rationale: Initial query is vague; rewrite to clarify
retrieve n_docs=8 (all local_security)
coverage score=0.09 missing=['definition', 'examples']
post     gap_retry     payload={"missing_aspects": ["definition", "examples"]}
pre      rewrite       (same rewrite again)
retrieve n_docs=8 (same docs)
coverage score=0.09 missing=['definition', 'examples']
post     gap_retry
…budget exhausted, answer
```

Two failure modes compound:

1. **The LLM policy rewrites every query, even when it is already clear.**
   The original "What is OWASP A03 injection?" is unambiguous. Rewriting
   it lengthened the query and *hurt* lexical retrieval. The DeepSeek
   pre-policy treated rewrite as the safe default, which is the wrong
   prior for our index.

2. **The aspect-coverage heuristic produced a false negative.**
   `compute_coverage` flagged `definition` as missing because none of the
   retrieved docs contained the literal word "define" / "definition". The
   docs *did* explain what SQL injection is. The post-policy correctly
   trusted the coverage signal and chose `gap_retry`, which then re-ran
   the same broken rewrite. After the retries the answer was based on
   evidence that didn't lexically match — so the judge gave 0.0
   factual_accuracy.

So: it's a **bad-prior LLM policy + brittle coverage heuristic** problem,
not a "we need RL" problem.

## Bug found during A-Mem spot-check

A separate run with A-Mem memory + DeepSeek over 2 turns surfaced a
probe bug: for the follow-up "And how do I prevent it?", the probe
picked `Based` (the first capitalized word of the prior answer, "Based
on the provided evidence…") as the coref target for `it`. Pre-policy
then rewrote "it" → "Based", which made the answer nonsensical.

Fix is small: in `memory/probe.py`, change the capitalized-noun pattern
to exclude common sentence-initial words ("Based", "The", "This", "I",
"It", "We") and prefer ALL-CAPS acronyms (OWASP, NIST, CVE) or longer
multi-word entities.

## What this means for "should we train RL"

**No, not yet.** RL would train a policy network to imitate or beat the
current LLM-policy. But the current LLM-policy is **already worse than
the rule policy** — there is nothing useful to imitate, and improving on
it does not require RL.

The actionable next steps are:

1. **Cheapest win**: ship the rule policy as the default. It is already
   the better policy on this benchmark.
2. **Easy upgrade**: tighten the `LLMPrePolicy` system prompt to prefer
   `noop` aggressively, and stop the LLM from rewriting unless there is
   a *specific* trigger (coref present, decomposable conjunction, etc.).
   Add the available providers list to the prompt so `choose_provider`
   isn't a dart throw.
3. **Higher leverage**: replace `compute_coverage` (lexical) with an
   LLM-as-judge coverage check. The same `LLMJudge` infra works; use a
   small focused prompt: "given Q and retrieved docs, list missing
   aspects". The current heuristic produces false positives that drive
   the wrong retries.
4. **Bug fix**: tighten probe coref extraction (5-minute change).
5. Only after these are done — and only if a measurable gap remains
   between LLM-policy and an oracle that picks the right action — should
   training enter the picture.

## Artifacts (committed)

```
experiments/baseline_2026_05_17/baseline_rule.jsonl   # 8 trajectories, RulePolicy
experiments/baseline_2026_05_17/baseline_llm.jsonl    # 8 trajectories, LLMPolicy (DeepSeek)
experiments/baseline_2026_05_17/scored_rule.jsonl     # judge verdicts
experiments/baseline_2026_05_17/scored_llm.jsonl
experiments/baseline_2026_05_17/amem_check.jsonl      # 2-turn A-Mem smoke
```

Reproduce:
```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com
export SA_LLM_BACKEND=openai SA_LLM_MODEL=deepseek-chat
export SA_RETRIEVER_BACKEND=handmade_rag
export SA_HANDMADE_RAG_ROOT=/path/to/all-in-rag/code/handmade_rag
export SA_MEMORY_BACKEND=inmem SA_LLM_MAX_TOKENS=600 SA_MAX_RETRIES=2

python scripts/collect_trajectories.py --policy rule --out runs/baseline_rule.jsonl
python scripts/collect_trajectories.py --policy llm  --out runs/baseline_llm.jsonl
python scripts/score_trajectories.py runs/baseline_rule.jsonl --judge llm --out runs/scored_rule.jsonl
python scripts/score_trajectories.py runs/baseline_llm.jsonl  --judge llm --out runs/scored_llm.jsonl
```
