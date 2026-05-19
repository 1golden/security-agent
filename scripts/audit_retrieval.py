#!/usr/bin/env python3
"""Retrieval quality audit on an existing trajectory JSONL.

For each (question, retrieved_doc) pair, ask the LLM: is this doc
relevant to the question? Compute relevance@k and identify whether the
score ceiling is bound by retrieval quality vs generation quality.

Usage:
    python scripts/audit_retrieval.py runs/baseline_rule_v2.jsonl --out runs/audit.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from security_agent.config import Config                       # noqa: E402
from security_agent.generation.llm_generator import build_llm  # noqa: E402


_SYSTEM = (
    "You are a retrieval-quality auditor. Given a user question and ONE "
    "retrieved document chunk, decide whether the chunk is RELEVANT to "
    "answering that specific question.\n"
    "A relevant chunk directly addresses the topic the question is asking "
    "about. A tangentially-related chunk (same broad domain but different "
    "specific topic) is NOT relevant.\n"
    'Reply JSON ONLY: {"relevant": <0 or 1>, "reason": "<one line>"}'
)


def audit_one(llm, question: str, chunk_text: str) -> tuple[int, str]:
    user = f"Question:\n{question}\n\nChunk:\n{chunk_text[:1500]}"
    try:
        data = llm.json(_SYSTEM, user)
    except Exception as e:
        return 0, f"llm error: {e}"
    rel = int(bool(data.get("relevant", 0)))
    return rel, str(data.get("reason", ""))[:200]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("traj", help="trajectory JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--max-docs", type=int, default=8, help="per-question doc cap")
    args = p.parse_args()

    cfg = Config()
    llm = build_llm(cfg.llm)

    results: list[dict] = []
    with open(args.traj, encoding="utf-8") as fh:
        traj_lines = [ln for ln in fh if ln.strip()]

    for ti, line in enumerate(traj_lines):
        try:
            traj = json.loads(line)
        except json.JSONDecodeError:
            continue
        q_text = traj.get("question", {}).get("text", "")
        # collect last-seen doc texts (the final retrieve pass docs)
        last_docs: list[dict] = []
        for step in traj.get("steps", []):
            if step.get("stage") != "retrieve":
                continue
            obs = step.get("observation") or {}
            if obs.get("doc_texts"):
                last_docs = list(obs["doc_texts"])
        last_docs = last_docs[: args.max_docs]

        rels = []
        for di, d in enumerate(last_docs):
            rel, reason = audit_one(llm, q_text, d.get("text") or "")
            rels.append({
                "rank": di + 1,
                "doc_id": d.get("id"),
                "source": d.get("source"),
                "score": d.get("score"),
                "relevant": rel,
                "reason": reason,
            })
        n = len(rels)
        n_rel = sum(r["relevant"] for r in rels)
        prec1 = float(rels[0]["relevant"]) if rels else 0.0
        prec3 = sum(r["relevant"] for r in rels[:3]) / max(min(n, 3), 1) if rels else 0.0
        prec5 = sum(r["relevant"] for r in rels[:5]) / max(min(n, 5), 1) if rels else 0.0
        record = {
            "i": ti,
            "question": q_text,
            "n_docs": n,
            "n_relevant": n_rel,
            "precision_at_1": prec1,
            "precision_at_3": prec3,
            "precision_at_5": prec5,
            "per_doc": rels,
        }
        results.append(record)
        print(
            f"[{ti+1:>3}/{len(traj_lines)}] n={n} relevant={n_rel}  "
            f"P@1={prec1:.2f} P@3={prec3:.2f} P@5={prec5:.2f}  "
            f"Q: {q_text[:55]!r}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # aggregate
    if results:
        agg = {
            "n_questions": len(results),
            "mean_precision_at_1": sum(r["precision_at_1"] for r in results) / len(results),
            "mean_precision_at_3": sum(r["precision_at_3"] for r in results) / len(results),
            "mean_precision_at_5": sum(r["precision_at_5"] for r in results) / len(results),
            "mean_relevant_per_q": sum(r["n_relevant"] for r in results) / len(results),
        }
        print(json.dumps(agg, indent=2))
    print(f"\nWrote audit to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
