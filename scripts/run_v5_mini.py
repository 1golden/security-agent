#!/usr/bin/env python3
"""v5 mini-benchmark: run a hand-picked subset of the 50-Q SAQ over the
new security_index_v5 to verify the E2 (tactic sub-route) + E3 (NIST
800-53 ingest) fixes lift the affected questions before committing to
the full 3 × 50 = 150-Q run.

Subset = the 4 tactic Qs (i=0-3) plus a representative access-control
Q (i=29). Compares Rule-policy v5 against the v4 score per Q.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/mnt/d/Project/WORK/all-in-rag/code/handmade_rag/src")

from security_agent.config import Config
from security_agent.eval.judge import build_judge
from security_agent.eval.score_trajectories import _extract_evidence
from security_agent.generation.llm_generator import build_llm
from security_agent.pipeline import build_default_pipeline
from security_agent.types import Question

BENCH = Path("/mnt/d/Project/WORK/all-in-rag/data/benchmarks/security_benchmark_saq.jsonl")
V4_RULE = Path("/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_19_v4/bench50_rule_v4.jsonl")
OUT = Path("/mnt/d/Project/WORK/security-agent/experiments/baseline_2026_05_22_v5/mini_rule_v5.jsonl")
TARGET_IS = [0, 1, 2, 3, 29]

# Pull V4 scores for direct comparison
v4_scores = {}
for line in open(V4_RULE):
    d = json.loads(line)
    v4_scores[d["i"]] = (d["score"], d["category"], d.get("ground_truth", ""))

# Load bench
bench = [json.loads(l) for l in open(BENCH) if l.strip()]

cfg = Config()
cfg.retriever.backend = "handmade_rag"
cfg.retriever.handmade_rag_index_dir = "/mnt/d/Project/WORK/all-in-rag/code/handmade_rag/artifacts/security_index_v5"
cfg.pipeline.pre_policy = "rule"
cfg.pipeline.post_policy = "rule"
cfg.pipeline.write_policy = "rule"
# Cross-Q memory pollution caused the v5 mini-bench i=3 regression (the
# in-memory layer leaked prior-Q technique IDs into Q4's evidence pool,
# clobbering the new ID-direct retrieval). Disable memory for honest
# per-Q comparison.
# NOTE: 'none' falls through to InMemoryStore — there is no real disable
# switch. Cross-Q pollution is killed by clearing the memory between Qs
# below (pipeline.memory._records.clear()).
cfg.memory.backend = "inmem"

pipe = build_default_pipeline(cfg)
llm = build_llm(cfg.llm)
judge = build_judge("llm", llm=llm)

OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"running mini-bench on {len(TARGET_IS)} Qs (v4 → v5)")
print(f"  {'i':>3s} {'category':<22s} {'v4_score':>9s} {'v5_score':>9s} {'delta':>7s}")
print(f"  {'-'*3} {'-'*22} {'-'*9} {'-'*9} {'-'*7}")

rows = []
deltas = []
for i in TARGET_IS:
    rec = bench[i]
    q_text = rec["question"]
    gt = rec.get("answer", "")
    cat = rec.get("category", "")
    src = rec.get("source", "")
    q = Question(text=q_text, session_id=f"v5-mini-{i:03d}",
                 metadata={"ground_truth": gt, "category": cat, "source": src})
    # Clear the in-process memory store so prior Qs in this loop don't
    # leak token-overlap matches into this Q's memory search step. The
    # default InMemoryStore.search() ignores session_id and pulls from
    # a single global dict (memory/inmem.py:21) — silent cross-Q bias.
    mem = getattr(pipe, "memory", None) or getattr(pipe, "_memory", None)
    if mem is not None and hasattr(mem, "_records"):
        mem._records.clear()
        if hasattr(mem, "_by_session"):
            mem._by_session.clear()
    t0 = time.time()
    try:
        result = pipe.ask(q)
        traj = result.trajectory.to_dict()
        ev = _extract_evidence(traj)
        v = judge.score(q_text, result.answer or "", evidence=ev, ground_truth=gt)
        v5_score = v.score
        rows.append({"i": i, "question": q_text, "ground_truth": gt, "category": cat,
                     "source": src, "answer": result.answer or "",
                     "score": v5_score, "dimensions": v.dimensions,
                     "tokens": result.trajectory.metadata.get("tokens_spent", 0),
                     "elapsed": time.time() - t0})
        v4_s = v4_scores.get(i, (None, "?", ""))[0]
        delta = (v5_score - v4_s) if v4_s is not None else 0
        deltas.append(delta)
        print(f"  {i:>3d} {cat:<22s} {v4_s:>9.3f} {v5_score:>9.3f} {delta:>+7.3f}")
    except Exception as e:
        print(f"  {i:>3d} {cat:<22s}  ERROR: {type(e).__name__}: {e}")
        rows.append({"i": i, "error": f"{type(e).__name__}: {e}"})

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nwrote {OUT}")
if deltas:
    print(f"mean Δ = {sum(deltas)/len(deltas):+.3f}  (positive = v5 better)")
