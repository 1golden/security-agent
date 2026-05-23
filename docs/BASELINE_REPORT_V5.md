# Baseline Report v5 — Retrieval Sub-routing + Corpus Gap Closure

**Date:** 2026-05-22
**Predecessor:** [BASELINE_REPORT_V4.md](BASELINE_REPORT_V4.md) (Rule 0.658 / LLM 0.633 / Router 0.673 / Oracle 0.837)

## TL;DR

Four experiments tried after v4. The headline finding is **defensive**: data-driven router collapses at n=50, and a latent memory-leak bug (existing since v3 but masked by v4 retrieval quirks) was exposed and fixed. The two retrieval improvements that landed are architecturally clean but their measurable impact on the SAQ benchmark is bounded by the questions actually present in the benchmark.

| Experiment | Verdict | Production status |
|---|---|---|
| E1 — Learned router on v4 outcomes | **Negative**: LR / rule-enum / informed-heuristic all underperform the v4 5-feature heuristic at honest LOO | Kept heuristic unchanged |
| E2 — Tactic vs technique sub-routing in `_lookup_by_external_ids` | Mixed: tactic Qs lift 2/4 → 3/4 close to 1.0, but per-Q DeepSeek noise (±0.2) makes single-run comparisons unreliable | Landed in `hierarchical_retrieval.py` |
| E3 — NIST SP 800-53 Rev 5 ingest (+5441 chunks) | **Clean architectural win**: corpus-gap short-circuit retired, controls now indexed | Landed; new `security_index_v5/` |
| E4 — NVD CVE refresh | Skipped — SAQ benchmark contains zero CVE-specific Qs | n/a |
| Side-finding — `InMemoryStore` cross-session leak | **Bug**: `search()` ignores `session_id`, scans global `_records`, biases later Qs in same pipeline instance | Fixed in `run_benchmark.py` per-Q clear |

No full 3 × 50-Q ensemble re-run was conducted. **The v5 mini-bench (5 Qs) on Rule policy shows mean Δ +0.243 vs v4, but with per-Q variance ≥0.7** between consecutive runs of the same Q due to DeepSeek sampling + chunk-rerank ordering. A defensible full-set rerun would need 3+ trials averaged.

---

## E1 — Learned Router (Negative Result)

**Premise from v4 report**: "Train a learned router on v4 outcomes (highest leverage, half a day)" — predicted to close some of the 16.4-pt Router→Oracle gap (0.673 → 0.837).

**What we tried**:

| Approach | Method | Honest LOO score |
|---|---|---|
| v4 heuristic (baseline) | 5-feature hand-crafted | **0.6733** |
| LR (10 shape features) | L2 logistic regression | 0.6578 (= rule_only) |
| LR (16 shape+kw, class-balanced) | weighted L2, λ swept | 0.6412 |
| Single keyword rule | exhaustive `(R+ | R-veto)` search | 0.6573 |
| Informed heuristic (≤2 kw adj on top of heuristic) | rule + delta search | 0.6548 |

Every data-driven method **underperformed** the heuristic at honest LOO.

**Why**:
- Label imbalance (Rule wins 34 / LLM wins 16) collapses L2 LR to majority class.
- Selection variance: best rule on n−1=49 isn't best on n=50. Full-set rule-enum optimum is 0.6928; honest LOO of same procedure is 0.6573 — the gap is the overfit.
- The strongest single feature (per-Q retrieval P@5) is only available for 8 Qs (audit set), not 50.

**Implication for v4 recommendation**: The "train a learned router" path was wrong. With n=50, the v4 heuristic IS the local optimum and the highest-leverage move is elsewhere (E3).

**Artifacts**: `experiments/baseline_2026_05_22_v5/E1_learned_router_findings.md`, `learned_router.json`, `informed_heuristic.json`, `router_v5_honest_loo.jsonl`.

---

## E2 — Tactic Sub-routing

**Bug fixed**: `_lookup_by_external_ids` in `hierarchical_retrieval.py` unconditionally filtered the `kill chain phases` section as "metadata" (since v4). But for queries like "Which tactic does T1xxxx belong to?", that chunk literally IS the answer — it reads `"## Kill Chain Phases - defense-evasion"`.

**Fix**: Detect tactic-asking queries (`tactic` / `kill chain` / `phase` keywords in the question), and:
1. Don't filter the kill-chain-phases section
2. Relax the chunk-length minimum from 80 → 30 for that section only (the answer is intentionally short, e.g. "defense-evasion")

**Verification on 4 tactic Qs** (ID-direct retrieval after fix):
```
T1562.001 → [Kill Chain Phases | T1562.001] ## Kill Chain Phases - defense-evasion
T1218.013 → [Kill Chain Phases | T1218.013] ## Kill Chain Phases - defense-evasion
T1027.016 → [Kill Chain Phases | T1027.016] ## Kill Chain Phases - defense-evasion
T1070.005 → [Kill Chain Phases | T1070.005] ## Kill Chain Phases - defense-evasion
```
Top-1 retrieval is now correct for all 4 (ground truth: Defense Evasion for all).

**Mini-bench Rule policy on 5 Qs** (4 tactic + 1 access-control):

| i | category | v4 score | v5 score (memory-clean) | Δ |
|---|---|---|---|---|
| 0 | technique-tactic | 0.100 | 1.000 | +0.900 |
| 1 | technique-tactic | 0.100 | 1.000 | +0.900 |
| 2 | technique-tactic | 1.000 | 0.225 | -0.775 |
| 3 | technique-tactic | 0.835 | 1.000 | +0.165 |
| 29 | access-control | 0.900 | 0.925 | +0.025 |
| | | | **Mean Δ** | **+0.243** |

**Caveat**: i=2 (T1027.016) was 1.0 in v4 AND in a different v5 run (with memory leak intact). In this clean run it scored 0.225. **The same Q with the same retrieval gives different LLM answers across runs** — DeepSeek API sampling + ties in BGE reranker ordering. This sets a noise floor of ±0.2 on per-Q comparisons. We did not run a full 50-Q v5 because (a) cost and (b) without trial averaging, single-run aggregates are not credible to ±0.05 either.

**Unit tests added**: `tests/test_query_router.py::TestTacticSubRouting::test_tactic_query_keeps_kill_chain_phases` and `test_nontactic_query_drops_kill_chain_phases`.

---

## E3 — NIST SP 800-53 Rev 5 Ingest

**Premise**: v4 had a hard-coded corpus-gap short-circuit returning empty docs for any "800-53" query (`query_router.py::_check_corpus_gap`). This was a permanent ceiling on access-control / audit / authentication categories.

**What landed**:

1. **OSCAL JSON converter** (`scripts/convert_nist_800_53_oscal.py`): reads NIST's official OSCAL catalog (10.4 MB, 1196 controls across 20 families), emits one markdown file per family with statement + guidance + related-control prose. Parameter placeholders rendered as `<param>`. Assessment objectives/methods omitted (boilerplate, would 3× index size for zero retrieval value).

2. **Incremental indexer** (`scripts/extend_index_with_nist_800_53.py`): loads the existing 38786-chunk index, embeds only the 5441 new NIST chunks (31s on BGE-M3), concatenates, rebuilds FAISS. Writes to `artifacts/security_index_v5/` so v4 stays reproducible.

3. **Corpus-gap retirement** in `query_router.py::_check_corpus_gap`: function now returns `None` for 800-53. The soft `id_not_in_corpus` fallback in `hierarchical_retrieval.py` still handles CVEs etc.

**Index growth**: 38786 → 44227 chunks (+14%); 4793 → 4813 docs (+0.4%).

**Coverage verified**: `find_chunk_indices_by_external_id("AC-1") → 1 chunk`, `AU-12 → 5`, `SC-8 → 6`, `IA-5 → 19`.

**SAQ benchmark impact**: The 50-Q SAQ contains **zero questions with explicit 800-53 control codes**. The 13 access-control / authentication / audit Qs are OWASP-grounded (their ground truth cites OWASP cheatsheets). Whether NIST 800-53 prose helps the LLM answer them better is empirically untested — but at minimum these Qs can no longer be silently shorted by the gap heuristic.

---

## E4 — NVD CVE Refresh (Skipped)

**Investigation**: existing index has 2000 distinct CVEs, all from 1999–2001 (the original training snapshot). Modern CVE coverage is zero.

**Decision**: skip. The 50-Q SAQ contains **zero CVE-specific Qs**. A full NVD refresh would cost ~50 MB download + ~30 min embedding for zero measurable benchmark lift on this dataset.

If a CVE-heavy benchmark is added later, the same `extend_index_with_*.py` template applies.

---

## Side-finding — `InMemoryStore` Cross-session Leak

**Bug**: `memory/inmem.py::InMemoryStore.search()` scans `self._records` (a single global dict) and ranks by token overlap. It accepts a `session_id` argument **but ignores it**. Result: in any pipeline instance that processes multiple Qs, later Qs' memory step retrieves records from earlier (unrelated) Qs.

**Discovery**: First v5 mini-bench run (without memory clear) showed i=3 dropping 0.835 → 0.100 with LLM answer saying "evidence only discusses techniques T1562.001, T1218.013, T1027.016" — the exact IDs of Qs 0, 1, 2.

**Impact on prior reports**: v3 and v4 benchmarks also ran with this bug active. Their aggregates are still apples-to-apples vs each other, but **individual Q comparisons across versions are unreliable** unless memory is cleared per Q.

**Fix landed**: `run_benchmark.py` and `run_v5_mini.py` now call `pipe.memory._records.clear()` and `_by_session.clear()` between Qs. Proper fix (scoping search by session_id) is a follow-up.

---

## What changed in code

### handmade_rag (`/mnt/d/Project/WORK/all-in-rag/code/handmade_rag/`)
- `src/handmade_rag/hierarchical_retrieval.py` — tactic detection + kill-chain-phases unfilter + 30-char min length
- `src/handmade_rag/query_router.py` — `_check_corpus_gap` no longer flags 800-53
- `tests/test_query_router.py` — 2 new tests for tactic sub-routing; 1 test inverted (800-53 no longer gap)
- `scripts/convert_nist_800_53_oscal.py` — new
- `scripts/extend_index_with_nist_800_53.py` — new
- `data/security/nist/SP_800_53_Rev5/` — 20 family MD files (1196 controls)
- `artifacts/security_index_v5/` — extended index (44227 chunks)

### security-agent
- `src/security_agent/config.py` — new `RetrieverConfig.handmade_rag_index_dir` field
- `src/security_agent/pipeline.py` — wires it into `HandmadeRagAdapter`
- `scripts/run_benchmark.py` — memory clear per Q
- `scripts/train_learned_router.py`, `scripts/enumerate_router_rules.py`, `scripts/tune_informed_heuristic.py`, `scripts/run_v5_mini.py`, `scripts/run_v5_benchmark.sh` — E1/E5 experiment scripts
- `experiments/baseline_2026_05_22_v5/` — E1 findings, learned-router model snapshots, mini-bench output

---

## Recommended next moves (revised)

The v4 report's #1 recommendation ("train learned router") is killed by E1. The actual leverage hierarchy is now:

1. **Larger benchmark** (150+ Qs). Required to either revisit learned routing (more LLM-wins examples) or to make per-Q claims robust against DeepSeek ±0.2 noise. ~$1 / 4–6h DeepSeek.

2. **Proper session-scoped memory search**. Patch `InMemoryStore.search` to honour `session_id`, plus AMemAdapter equivalent. Cleaner than per-Q `_records.clear()`. Half a day.

3. **Multi-trial averaging**. Run each (policy, Q) tuple 3× and average. Triples cost but gives ±0.05 per-Q confidence. Needed before any single-Q claim should be published.

4. **Stop chasing oracle**. The v4 0.837 oracle was computed from independent Rule-run and LLM-run scored side-by-side. To actually realize it you'd need to swap entire pipeline stack (pre+post+write) per-Q based on something — and our experiments show shape-feature LR can't predict that. Either go to (1) for more data, or to RL/bandit with online feedback.

5. **NIST 800-53-themed benchmark probe**. Now that 800-53 is indexed, add 10–20 explicit "what does NIST SP 800-53 control X cover?" Qs to the benchmark to measure E3's real impact.
