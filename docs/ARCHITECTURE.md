# security-agent architecture

## The pipeline at a glance

```
Question
  │
  ├─→ Input Safety (wrapper)
  │
  ↓
Fast Memory Probe              (cheap signals: recent turns, filters,
  │                              coref, KG seeds, provider preferences)
  ↓
Pre-Retrieval Policy           (noop / rewrite / decompose /
  │                              inherit_filter / choose_provider)
  ↓
Memory Expansion               (KG multi-hop, vector hits, doc boost,
  │                              query expansion terms)
  ↓
RAG Retrieval + Rerank
  ↓
Coverage Check                 (returns {score, missing_aspects,
  │                              low_confidence_claims, redundancy, diagnosis})
  ↓
Post-Retrieval Policy          (filter / gap_retry / broaden / narrow /
  │                              stop / answer)
  │   ↑ retry under Budget {max_retries, max_tokens, min_coverage_delta}
  │   └─── back to Pre-Retrieval Policy
  ↓
Answer Generation
  ↓
Memory Write Policy            (per-axis: facts / constraints / aliases /
  │                              reflection / provider_utility)
  ↓
Memory Write + Eval Log ───────► (provider utility feeds back to Probe)
  │
  ↓
Output Safety (wrapper)
```

## Why these splits

### 1. Probe vs Expansion

The planner needs memory signals to decide what to retrieve, but reading
memory well requires a refined query — a chicken-and-egg loop. We split
memory into:

- **Probe** (cheap): recent turns, explicit filter detection, coref hints,
  KG seed nodes, learned provider preferences. No embedding, no LLM.
- **Expansion** (expensive): runs *after* the Pre-Retrieval Policy has
  rewritten / decomposed / chosen providers. Does the real vector +
  multi-hop work.

This mirrors the "core memory" + "archival memory" split used in
production systems like MIRIX and MemoryOS.

### 2. Pre vs Post Policy (symmetric, both swappable)

- **Pre** = planning (what to look for)
- **Post** = verification (did we find what we needed)

Splitting them lets the trajectory log attribute rewards to each
independently, which is essential for RL fine-tuning later (AgeMem-style
three-stage GRPO needs clean per-stage credit assignment).

### 3. Structured Coverage, not a single float

`coverage_score` alone cannot distinguish "we didn't get any mitigation
info" (→ gap_retry) from "we got 5 nearly-identical docs" (→ narrow) from
"score is low and no specific gap" (→ broaden). The `Coverage` dataclass
carries `missing_aspects`, `low_confidence_claims`, `redundancy`, and a
`diagnosis` summary so the Post-Retrieval Policy can pick the right
action.

### 4. Multi-axis Memory Write

`MemoryWritePolicy` is not a binary "write / skip". It decides per axis:

| Axis | Risk | Default in RuleWritePolicy |
|------|------|----------------------------|
| `facts` | low | on when coverage ≥ 0.4 |
| `constraints` | medium | on if imperative-mood markers in answer |
| `aliases` | **high** (MINJA poisoning vector) | off unless coref *and* coverage ≥ 0.6 |
| `reflection` | high cost | off unless idle (no retries this turn) and ≥ 8 prior turns |
| `provider_utility` | low | on when provider was filtered and coverage decent |

Aliases and reflection are gated because A-MemGuard 2025 showed they are
the most exploitable axes. RL training data will let the policy learn
better thresholds, but the rule defaults are conservative.

### 5. Budget at the post-loop

`Budget` carries `max_retries`, `max_tokens`, and `min_coverage_delta`.
A retry is allowed only when:

- attempts < max_retries
- cumulative LLM tokens < max_tokens
- the previous retry actually improved coverage by ≥ min_coverage_delta

The third condition is the one most architectures miss; without it a
broken policy can loop forever.

### 6. Closed-loop provider preference

The `MemoryWritePolicy` writes `provider_utility` when retrieval used a
specific provider and coverage was decent. The next session's
`FastMemoryProbe` reads this back via `MemoryStore.preferred_providers()`
and surfaces it as `probe.preferred_providers`, which the Pre-Retrieval
Policy can act on with `CHOOSE_PROVIDER`. This is the only feedback
edge in the diagram; everything else is forward-flow.

### 7. Safety as wrappers, not policies

Input/output safety run *outside* the policy layer so they cannot be
disabled by a bad pre/post decision. Default uses crude pattern matching;
Llama Guard / ShieldGemma / NeMo Guardrails can be swapped in by
subclassing `_PatternFilter`.

## File map

```
src/security_agent/
├── pipeline.py            # orchestrator (the while loop)
├── cli.py                 # `security-agent ask "..."`
├── types.py               # Question / State / Action / Coverage / Trajectory / Budget
├── config.py              # env-driven dataclass config
├── coverage.py            # heuristic Coverage.compute (LLM-based pluggable)
│
├── safety/wrappers.py     # InputSafety / OutputSafety
│
├── memory/
│   ├── base.py            # MemoryStore interface
│   ├── inmem.py           # zero-dep in-process store (default)
│   ├── amem_adapter.py    # wraps all-in-rag/A-mem-sys
│   ├── probe.py           # FastMemoryProbe (cheap)
│   └── expansion.py       # MemoryExpansion (heavy)
│
├── retrieval/
│   ├── base.py            # Retriever interface
│   ├── dummy.py           # in-process corpus, for tests/demo
│   └── handmade_rag_adapter.py  # wraps all-in-rag/code/handmade_rag
│
├── generation/
│   └── llm_generator.py   # LLMBackend (Dummy / OpenAI) + LLMGenerator
│
├── policies/
│   ├── base.py            # Policy / PrePolicy / PostPolicy / MemoryWritePolicy
│   ├── pre_retrieval/{rule,llm,rl}.py
│   ├── post_retrieval/{rule,llm,rl}.py
│   └── memory_write/{rule,llm,rl}.py
│
└── eval_log/
    ├── logger.py          # JSONL trajectory writer
    └── replay.py          # iterators for RL training data
```

## Trajectory log format

One JSONL line per episode. Each line is:

```json
{
  "question": {"text": "...", "session_id": "...", "user_id": null, ...},
  "steps": [
    {"step": 1, "stage": "input_safety", "state_before": {...}, "action": null,
     "observation": {"allow": true, "reason": ""}},
    {"step": 2, "stage": "probe", ...},
    {"step": 3, "stage": "pre", "action": {"type": "inherit_filter", ...}, ...},
    ...
  ],
  "final_answer": "...",
  "reward": null,
  "metadata": {}
}
```

For RL training:
- `(state_before, action)` pairs at `stage == "pre" | "post" | "write"`
  are the supervision units for behavior cloning warm-start.
- `reward` is left null at write time; downstream evaluators (LLM-as-judge
  on the final answer, plus per-step shaping) backfill it.

## Extension points (what to plug in for production)

| Slot | Default | Swap to |
|------|---------|---------|
| `LLMBackend` | `DummyLLM` | `OpenAILLM(cfg)` or your own subclass |
| `MemoryStore` | `InMemoryStore` | `AMemAdapter(amem_root)` |
| `Retriever` | `DummyRetriever` | `HandmadeRagAdapter(root)` |
| `InputSafety` / `OutputSafety` | pattern matcher | Llama Guard / ShieldGemma |
| `compute_coverage` | lexical heuristics | LLM-as-judge coverage |
| `RulePrePolicy` etc. | hand-written | LLMPrePolicy or RLPrePolicy |

All slots are constructor parameters of `Pipeline`; nothing about the
flow needs to change to swap any of them.
