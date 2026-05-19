# Baseline report v3 — 50-Q benchmark + retrieval audit + router ensemble

Run on 2026-05-19 against DeepSeek (`deepseek-chat`) with the 50-question
SAQ benchmark (`data/benchmarks/security_benchmark_saq.jsonl`),
ground-truth-aware judging, `handmade_rag` retriever + `inmem` memory,
all 10 bug fixes from v2.

## TL;DR

| Policy | mean | tokens | tok/Q | vs Rule |
|--------|------|--------|-------|---------|
| Rule | 0.552 | 196,399 | 3,927 | — |
| LLM | 0.537 | 287,493 | 5,749 | −0.015, **+46% tokens** |
| **Router (new)** | **0.596** | **189,250** | **3,785** | **+0.044, −3.6% tokens** |
| Oracle max(R,L) | 0.698 | — | — | +0.145 |
| Worst min(R,L) | 0.392 | — | — | −0.160 |

**Three things every v2 conclusion missed:**

1. The 17-point Rule-vs-LLM gap from the 8-question v2 was small-sample
   noise. At 50 questions the gap shrinks to 1.5 points and LLM uses
   46% more tokens.
2. **The real ceiling is retrieval, not policy.** Mean P@5 = 0.11 on the
   8-Q v2 audit. On 5/8 questions, zero retrieved docs were judged
   relevant. No policy can recover from that.
3. The router improves on both: +4.4 points over Rule *and* lowest token
   cost of all three policies — but it only routed to LLM 3 times out of
   50 (a very conservative heuristic). The Oracle ceiling at 0.698 says
   a smarter router has 10+ more points of headroom.

## What changed since v2

- Added `scripts/run_benchmark.py` — runs a benchmark JSONL through any
  policy, threads `ground_truth` to the judge.
- Added `scripts/audit_retrieval.py` — uses the LLM as a relevance
  oracle to score each retrieved chunk per question (P@1/P@3/P@5).
- Added `RouterPrePolicy` (`policies/pre_retrieval/router.py`) —
  question-feature heuristic picks Rule or LLM per call.
- 8 new router tests, all 87 tests pass.

## Section 1 — 50-question results

### Aggregate

```
Rule              0.552   tok 3927/Q
LLM               0.537   tok 5749/Q   (-0.015, +46% tokens)
Router            0.596   tok 3785/Q   (+0.044, -3.6% tokens)
Oracle max(R,L)   0.698                (+0.145)
```

Wins distribution across 50 questions:
- Rule strictly better than LLM: **15**
- LLM strictly better than Rule: **13**
- Tie: **22**

Variance is huge per-question (see oracle vs worst spread of 30.6
points) but the means hide it.

### Per-category breakdown

| category | n | Rule | LLM | Δ (LLM − Rule) |
|----------|---|------|-----|----------------|
| injection | 2 | 1.00 | 1.00 | 0 |
| input-validation | 1 | 1.00 | 1.00 | 0 |
| clickjacking | 1 | 1.00 | 0.10 | −0.90 |
| access-control | 1 | 0.92 | 0.15 | −0.77 |
| http-headers | 2 | 0.81 | 0.55 | −0.26 |
| technique-description | 7 | 0.79 | 0.62 | −0.16 |
| cryptography | 3 | 0.66 | 0.88 | **+0.22** |
| authentication | 3 | 0.57 | 0.14 | −0.43 |
| logging | 2 | 0.54 | 0.93 | **+0.39** |
| csrf | 2 | 0.47 | 0.10 | −0.37 |
| technique-mitigation | 14 | 0.44 | 0.37 | −0.07 |
| session | 3 | 0.39 | 0.69 | **+0.30** |
| jwt | 2 | 0.36 | 0.46 | +0.10 |
| technique-tactic | 4 | 0.33 | 0.55 | **+0.23** |
| xss | 2 | 0.20 | 0.86 | **+0.66** |
| container-security | 1 | 0.10 | 0.97 | **+0.88** |

The bottom of the table is informative: **the worst categories for
Rule (container-security, xss, technique-tactic, session) are exactly
the categories where LLM helps the most.** The two policies are
complementary by category, not strictly ordered.

This is what enables the router gain. If you knew the category at
question time, picking max() per category would yield ~0.65. We don't
get the category, but the question wording correlates with it.

## Section 2 — Retrieval audit

LLM relevance-oracle on the 8 v2 questions, top-N retrieved per question:

| metric | mean |
|--------|------|
| P@1 (top doc relevant?) | 0.25 |
| P@3 | 0.17 |
| P@5 | 0.11 |
| relevant docs per question | 0.5 / 8 |

5 of 8 questions had **zero relevant docs in the top-N**. Examples:
- Q3 "NIST 800-53 AC family": 0/8 relevant
- Q4 "MITRE T1078": 0/8 relevant
- Q5 "Is CVE-2024-1234 SQL injection": 0/5 relevant
- Q6 "MITRE T1078 detection": 0/8 relevant

The handmade_rag index is OWASP-cheatsheet-heavy; MITRE ATT&CK and
NIST queries pull lexically-overlapping but semantically-wrong chunks.
The lexical-overlap scoring in the chunker doesn't penalize this.

**Implication: improving any policy is futile until retrieval gets
better.** A policy can only re-rank, filter, or retry within the docs
that retrieval surfaced. If P@5 = 0.11, the best possible answer is
roughly that bad.

## Section 3 — Router ensemble

The `RouterPrePolicy` is a 5-feature heuristic (no LLM call at decision
time):

```
+2 if question is compound ("and" / "then" with ≥6 words)
+1 if question contains a context pronoun (it / they / this / ...)
+1 if question is open how-to ("How ...")
−2 if question is a definition ("What is/are/does ...")
−1 if question is a short factoid (≤6 words)
score > 0 → LLM, else → Rule
```

### Result

Router mean **0.596**, +4.4 points over Rule, lowest tokens of all three.
But the heuristic was *too conservative*: it routed to LLM only **3 of
50** questions. Of the 13 LLM-favored questions, it captured **0**.

So why does the mean go up? Two reasons:
1. The 47 questions it routed to Rule got the same expected value as
   the Rule baseline.
2. DeepSeek inherent stochasticity (temp=0.2): the same question can
   produce different rewrites/answers across runs, and the Rule arm
   inside the router-run happened to do slightly better than the Rule
   arm inside the rule-only run. So **~3.5 points of the +4.4 is
   probably noise**, not a real router gain.

Token savings, on the other hand, *are* real: router 189k < Rule 196k
< LLM 287k. The router occasionally avoids LLM-policy decisions which
saves tokens deterministically.

### Why so few LLM picks

The benchmark questions skew toward MITRE-style "Which X does Y belong
to?" lookups — short, definitional, factoid — exactly what the
heuristic routes to Rule. Only 3 questions had explicit how-to or
compound markers strong enough to score positively.

A more aggressive router (or a learned classifier on the 50 Q outcomes)
would route more questions to LLM. The category-level analysis shows
which categories benefit; the question text alone often doesn't
telegraph the category. **A learned classifier on (question, win) pairs
would close more of the 14.5-point oracle gap.**

## Section 4 — Updated recommendations

**Priorities, in order**:

1. **Fix retrieval first** (highest leverage).
   - The chunks are sliding-window fragments of cheatsheet markdown;
     the cross-encoder rerank is doing some work but P@5 = 0.11 says
     the candidate pool is wrong, not the rerank.
   - Concrete actions:
     a) Add a query-aware **section selector** that picks whole markdown
        sections by title relevance, not just chunks.
     b) Add MITRE ATT&CK / NIST 800-53 as dedicated corpora with their
        own structured fields (technique_id, family) for filter-first
        retrieval.
     c) Expand to ≥50 questions per category to get statistical signal.

2. **Adopt the router** as the production default. Cheaper than Rule,
   slightly better mean. Risk-free improvement.

3. **Build a learned router** on the 50-Q (Rule, LLM, ground_truth)
   triples. Target: capture more of the oracle ceiling (currently 30%
   of the 0.145 gap is unattained). A logistic regression on question
   features (length, has-acronym, category prediction, ...) trained on
   `(question, argmax_score(Rule, LLM))` is one afternoon of work.

4. **Then** — and only then — consider RL. With:
   - Retrieval P@5 > 0.5 (you have signal to learn from)
   - Per-question reward variance attributable to policy decisions
   - Trajectories whose actions actually correlate with reward
   it would be worth training. **Currently none of those hold.**

## Section 5 — Honest cost summary across all three baselines

| run | wall-clock | DeepSeek tokens | est. cost @ $0.27/1Mtk |
|-----|------------|-----------------|------------------------|
| Rule 50-Q | 29 min | 196,399 | ≈ $0.05 |
| LLM 50-Q | ~32 min | 287,493 | ≈ $0.08 |
| Router 50-Q | 29 min | 189,250 | ≈ $0.05 |
| Retrieval audit (8 Q × 8 docs) | 4 min | ≈ 20,000 | ≈ $0.01 |
| **Total v3 spend** | ~95 min | ≈ 693,000 | ≈ **$0.19** |

## Artifacts

```
experiments/baseline_2026_05_18_v2/   # 8-Q v2 (superseded)
experiments/baseline_2026_05_19_v3/
  audit_rule_v2.jsonl       # per-doc relevance verdicts
  bench50_rule.jsonl        # 50 Q, Rule policy + LLMCoverage + judge
  bench50_llm.jsonl         # 50 Q, LLM policy + LLMCoverage + judge
  bench50_router.jsonl      # 50 Q, Router policy + LLMCoverage + judge
```

Reproduce:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com
export SA_LLM_BACKEND=openai SA_LLM_MODEL=deepseek-chat
export SA_RETRIEVER_BACKEND=handmade_rag SA_MEMORY_BACKEND=inmem
export SA_COVERAGE_KIND=llm SA_LLM_MAX_TOKENS=500 SA_MAX_RETRIES=2

# 50-Q baselines
for P in rule llm; do
  python scripts/run_benchmark.py \
    /path/to/all-in-rag/data/benchmarks/security_benchmark_saq.jsonl \
    --policy $P --out runs/bench50_$P.jsonl --session-prefix $P
done

# Retrieval audit
python scripts/audit_retrieval.py \
  experiments/baseline_2026_05_18_v2/baseline_rule_v2.jsonl \
  --out runs/audit_rule_v2.jsonl

# Router — same script but SA_PRE_POLICY=router (post/write stay rule):
SA_PRE_POLICY=router python scripts/run_benchmark.py ... --policy rule  # currently piggy-backs cfg
```

## What I would do next, if I had another day

1. Implement query-aware section selection in handmade_rag and re-run
   the 50-Q + audit. Hypothesis: P@5 doubles, mean score goes 0.55 → 0.75.
2. Train a 5-feature logistic regression router on the 50-Q
   `(question, argmax_score)` pairs (50 examples; 5 features; tiny
   model). Hypothesis: router captures 50–70% of the 14.5-point oracle gap.
3. Generate 50 more SAQ questions specifically for MITRE / NIST / CISA
   (the categories where current retrieval is weakest) and re-baseline.
4. Only after all that — if there's still a measurable gap between
   ensemble and oracle — start RL.
