"""Score a trajectory JSONL produced by the pipeline.

Two entry points:

- `score_trajectory_dict(traj, judge)` — score one in-memory trajectory
- `score_jsonl_file(path, judge)` — iterate a .jsonl, return list of (question, score)

The aggregate is what RL training uses as final reward; per-dim scores can
inform shaping (e.g., a low evidence_grounding hits the post-policy harder
than it hits the pre-policy).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from security_agent.eval.judge import Judge, JudgeVerdict


def _extract_evidence(traj: dict) -> list[dict]:
    """Walk the trajectory and reconstruct the evidence visible to the judge.

    Bug #1 fix: the pipeline now writes truncated doc text into the retrieve
    observation (`doc_texts`), so we can rebuild real evidence for the judge.
    Old-format trajectories (just `doc_ids`) still work — text just falls
    back to empty, which mirrors the old behavior.

    De-duplicates by doc id across retry passes; keeps the longest text seen.
    """
    by_id: dict[str, dict] = {}
    for step in traj.get("steps", []):
        if step.get("stage") != "retrieve":
            continue
        obs = step.get("observation") or {}
        # preferred path: rich doc_texts list
        for doc in obs.get("doc_texts") or []:
            did = str(doc.get("id"))
            if not did:
                continue
            text = str(doc.get("text") or "")
            existing = by_id.get(did)
            if existing is None or len(text) > len(existing.get("text", "")):
                by_id[did] = {
                    "id": did,
                    "text": text,
                    "source": doc.get("source"),
                    "score": doc.get("score"),
                }
        # legacy fallback: only doc_ids available
        if "doc_texts" not in obs:
            sources = obs.get("sources") or []
            for i, did in enumerate(obs.get("doc_ids", []) or []):
                did = str(did)
                if did not in by_id:
                    by_id[did] = {
                        "id": did,
                        "text": "",
                        "source": sources[i] if i < len(sources) else None,
                    }
    return list(by_id.values())


def score_trajectory_dict(traj: dict, judge: Judge) -> JudgeVerdict:
    q = traj.get("question", {}).get("text") or ""
    a = traj.get("final_answer") or ""
    ev = _extract_evidence(traj)
    return judge.score(q, a, evidence=ev)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def score_jsonl_file(path: str | Path, judge: Judge) -> list[dict]:
    """Score every trajectory in a JSONL. Returns rows for downstream stats."""
    rows: list[dict] = []
    for traj in iter_jsonl(path):
        verdict = score_trajectory_dict(traj, judge)
        rows.append(
            {
                "question": traj.get("question", {}).get("text", ""),
                "answer": traj.get("final_answer") or "",
                "score": verdict.score,
                "dimensions": verdict.dimensions,
                "reasoning": verdict.reasoning,
                "blocked": traj.get("metadata", {}).get("blocked", False),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    """Aggregate stats over a list of scored rows."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    avg = sum(r["score"] for r in rows) / n
    dim_avgs = {}
    for k in ("factual_accuracy", "evidence_grounding", "completeness", "safety"):
        vals = [r["dimensions"].get(k, 0.0) for r in rows]
        dim_avgs[k] = sum(vals) / n
    blocked = sum(1 for r in rows if r.get("blocked"))
    return {
        "n": n,
        "mean_score": avg,
        "mean_dimensions": dim_avgs,
        "n_blocked": blocked,
    }
