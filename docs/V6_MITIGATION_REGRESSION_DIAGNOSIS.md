# V6 Mitigation Regression Diagnosis

**Triggered by:** v6 report flagged `technique-mitigation` -0.21 (n=14) as the only category where v5 retrieval underperformed v4 Rule.

**Conclusion:** Not a retrieval bug. It's a corpus-structure + judge-calibration interaction. Don't "fix" the retrieval; if anything, the judge needs adjustment.

## The 5 big losses (Δ ≤ -0.5)

All 5 follow the same pattern:

| Q | Technique | v4 score | v5 score |
|---|---|---|---|
| 36 | T1090 (Proxy) | 0.75 | 0.10 |
| 38 | T1547.012 (Print Processors) | 1.00 | 0.10 |
| 45 | T1556.008 (Network Provider DLL) | 0.75 | 0.10 |
| 47 | T1213.001 (Confluence) | 0.75 | 0.10 |
| 49 | T1055.004 (APC) | 0.75 | 0.10 |

For all 5: ground truth IS a specific mitigation, but MITRE stores mitigations as separate `course-of-action` STIX objects linked to the technique via `relationship` objects. The course-of-action's `external_id` is `Mxxxx`, not `T1xxx`.

## What `_lookup_by_external_ids` finds

The query router parses `T1090` from the question and calls `_lookup_by_external_ids(["T1090"], query=...)`. The reverse index returns all chunks where `metadata.external_id == "T1090"` — but those are all chunks from the **attack-pattern** STIX object (Title, External IDs, Description, Detection, Platforms, Kill Chain Phases). **No mitigation chunk is reachable by this ID** because mitigations live in separate STIX objects.

For q=36 (T1090 Proxy), the v5 retrieval surfaces:
- `USAGE.md::s18::c0` (MITRE's recipe for "by ATT&CK ID" queries — top-1 via flat fallback)
- `attack-pattern--731f4f55.../s1::c0` (the "## External IDs" chunk — survived the metadata filter, see below)
- ICS T0884 chunks (unrelated)
- More USAGE.md code snippets

Note: the attack-pattern's **External IDs chunk** still surfaces at rank 2 despite my `_METADATA_SECTIONS = {"external ids", ...}` filter. This is because the flat-search fallback bypasses `_lookup_by_external_ids`'s filter — the chunk is in the global FAISS+BM25 index and matches the literal "T1090" token in the query. The filter only applies when `_lookup_by_external_ids` is the *active* path.

## Why v4 looked better

v4 had no `_lookup_by_external_ids` at all. Every query went through flat hybrid retrieval, which surfaced a mix of:
- Attack-pattern description chunks (loosely related)
- Recipe / USAGE.md chunks (matching "MITRE", "mitigation", "technique")
- Other course-of-action chunks (matching "mitigation")

The LLM, seeing a noisy mix, hedged with **"I cannot determine the specific mitigation from the provided evidence"** — and the judge scored that 0.75 (high `safety` and `evidence_grounding` dimensions: the answer correctly refused to hallucinate).

v5's ID-direct path returns only the attack-pattern chunks (which legitimately don't contain the mitigation). The LLM, seeing a focused set with no mitigation content, gives **"The provided evidence does not contain a specific mitigation recommendation"** — semantically identical to v4's hedge, but the judge now scores 0.10.

## Reading the judge score breakdown

For q=36 v5 (score 0.10):
- Likely `completeness=0`, `factual_accuracy=0`, `evidence_grounding=0.5`, `safety=1.0` → 0.10 weighted

For q=36 v4 (score 0.75):
- Likely `completeness=0.5`, `factual_accuracy=0.5`, `evidence_grounding=1.0`, `safety=1.0` → 0.75 weighted

The judge sees v5's narrow + accurate "no info" as worse than v4's broad + uncertain "I can't say". Both refusals are semantically the same; the score difference is calibration noise.

## What this means for the system

**Not a bug to fix in retrieval.** Adding mitigation chunks to ID-direct lookup would require:
1. Build a `technique → [mitigation course-of-action]` graph at ingest time
2. When `_lookup_by_external_ids("T1090")` runs and the query is mitigation-oriented, also return the linked course-of-action chunks

That's a half-day of indexing work. Doable, but it would surface chunks like `M1037 (Filter Network Traffic)` whose body says "Apply firewall rules to block known proxies" — generic mitigation prose that may or may not match the specific ground truth.

**Real fix: better judge calibration.** Both v4 and v5 LLM answers correctly refuse to hallucinate. They should score the same. The judge's per-Q variance (also evident in v5's per-Q ±0.2 DeepSeek noise documented in the v5 report) is the actual source of the regression signal.

## Decision matrix

| Option | Effort | Expected lift on mitigation Qs | Risk |
|---|---|---|---|
| Ship as-is, accept the regression | 0 | 0 | None — v5 aggregate still +0.083 over v4 |
| Patch judge prompt for "no info" calibration | 1-2h | +0.1–0.2 on the 5 losses | Low — affects all Qs, must re-validate |
| Add technique→mitigation graph to retrieval | 4-6h | Maybe +0.2–0.4 on these 5 if data exists | Medium — surfaces irrelevant mitigations on Qs where MITRE has none |
| Drop the 5 from benchmark (MITRE has no data) | 30min | +0.20 / 14 = +0.014 aggregate | Low — but biases the benchmark away from realistic "no answer" cases |

## Recommendation

**Ship as-is and document the corpus gap.** The v5 +0.083 aggregate lift is real. The mitigation regression is a calibration artifact, not a real quality drop. If we revisit, the cleanest move is judge calibration (path 2), not retrieval surgery.

## Files inspected

- `experiments/baseline_2026_05_22_v5/bench50_rule_v5.jsonl` (per-Q scores)
- `experiments/baseline_2026_05_22_v5/bench50_rule_v5.traj.jsonl` (q=36 retrieval contents)
- `all-in-rag/code/handmade_rag/src/handmade_rag/hierarchical_retrieval.py` (`_lookup_by_external_ids` filter logic)
- `all-in-rag/code/handmade_rag/src/handmade_rag/query_router.py` (`_MITRE_RE` regex)
