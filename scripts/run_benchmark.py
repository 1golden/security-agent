#!/usr/bin/env python3
"""Run the pipeline on a benchmark JSONL with ground-truth answers.

Each line in the input file is a record with at least:
    {"question": "...", "answer": "ground truth ...", "source": "...",
     "category": "..."}

For each record:
  1. pipe.ask() to get trajectory + answer
  2. LLMJudge.score(question, answer, evidence, ground_truth=answer)
     so the judge can compare against the reference
  3. Write one combined record per question to OUT.jsonl

Final summary prints aggregate + per-category breakdown.

Usage:
    python scripts/run_benchmark.py BENCHMARK.jsonl --policy rule --out runs/foo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from security_agent.config import Config                              # noqa: E402
from security_agent.eval.judge import build_judge                      # noqa: E402
from security_agent.eval_log.logger import TrajectoryLogger            # noqa: E402
from security_agent.eval.score_trajectories import _extract_evidence   # noqa: E402
from security_agent.generation.llm_generator import build_llm          # noqa: E402
from security_agent.pipeline import build_default_pipeline             # noqa: E402
from security_agent.types import Question                              # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", help="path to benchmark JSONL")
    p.add_argument("--policy", choices=["rule", "llm", "router"], default="rule")
    p.add_argument("--out", required=True, help="output JSONL with per-Q scoring")
    p.add_argument("--limit", type=int, default=None, help="cap to first N questions")
    p.add_argument("--session-prefix", default="bench")
    p.add_argument("--judge", choices=["llm", "heuristic"], default="llm")
    p.add_argument("--no-judge", action="store_true", help="skip judging entirely")
    args = p.parse_args()

    cfg = Config()
    cfg.pipeline.pre_policy = args.policy
    # Router only swaps the pre policy; post/write stay on rule so we measure
    # ensemble effect at the rewrite stage alone, not a stacked change.
    if args.policy == "router":
        cfg.pipeline.post_policy = "rule"
        cfg.pipeline.write_policy = "rule"
    else:
        cfg.pipeline.post_policy = args.policy
        cfg.pipeline.write_policy = args.policy

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # trajectories side-channel (full JSONL); per-Q scored rows in --out
    traj_path = out.with_suffix(".traj.jsonl")
    traj_logger = TrajectoryLogger(log_dir=str(traj_path.parent), filename=traj_path.name)
    pipe = build_default_pipeline(cfg, logger=traj_logger)

    llm = build_llm(cfg.llm) if (not args.no_judge and args.judge == "llm") else None
    judge = None
    if not args.no_judge:
        judge = build_judge(args.judge, llm=llm) if args.judge == "llm" else build_judge("heuristic")

    rows: list[dict] = []
    questions = []
    with open(args.benchmark, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.limit:
        questions = questions[: args.limit]

    t0 = time.time()
    for i, rec in enumerate(questions):
        q_text = rec.get("question", "").strip()
        gt = rec.get("answer", "")
        category = rec.get("category", "")
        source = rec.get("source", "")
        if not q_text:
            continue
        question = Question(
            text=q_text,
            session_id=f"{args.session_prefix}-{i:03d}",
            metadata={"ground_truth": gt, "category": category, "source": source},
        )
        try:
            result = pipe.ask(question)
        except Exception as e:
            print(f"[{i+1:>3}/{len(questions)}] ERROR: {type(e).__name__}: {e}", flush=True)
            rows.append({"i": i, "question": q_text, "ground_truth": gt,
                          "category": category, "source": source,
                          "answer": "", "error": str(e)})
            continue

        # judge with ground truth
        if judge is not None:
            traj_dict = result.trajectory.to_dict()
            evidence = _extract_evidence(traj_dict)
            verdict = judge.score(q_text, result.answer or "", evidence=evidence, ground_truth=gt)
            score = verdict.score
            dims = verdict.dimensions
        else:
            score = None
            dims = {}

        row = {
            "i": i,
            "question": q_text,
            "ground_truth": gt,
            "category": category,
            "source": source,
            "answer": result.answer or "",
            "blocked": result.blocked,
            "score": score,
            "dimensions": dims,
            "tokens": result.trajectory.metadata.get("tokens_spent", 0),
        }
        rows.append(row)

        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (len(questions) - (i + 1)) / max(rate, 1e-6)
        score_str = f"{score:.2f}" if score is not None else "-"
        print(
            f"[{i+1:>3}/{len(questions)}] score={score_str} "
            f"tok={row['tokens']} cat={category[:18]}  Q: {q_text[:55]!r}  "
            f"eta {eta:.0f}s",
            flush=True,
        )

    # write per-Q rows
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    traj_logger.close()

    # summary
    scored = [r for r in rows if r.get("score") is not None]
    if scored:
        agg = sum(r["score"] for r in scored) / len(scored)
        dims_keys = sorted({k for r in scored for k in r["dimensions"].keys()})
        dim_avgs = {
            k: sum(r["dimensions"].get(k, 0.0) for r in scored) / len(scored)
            for k in dims_keys
        }
        tokens_total = sum(r.get("tokens", 0) for r in scored)
        by_cat: dict[str, list[float]] = {}
        for r in scored:
            by_cat.setdefault(r["category"] or "unknown", []).append(r["score"])
        per_cat = {k: sum(v) / len(v) for k, v in by_cat.items()}
        print(json.dumps(
            {
                "n": len(rows),
                "n_scored": len(scored),
                "mean_score": agg,
                "mean_dimensions": dim_avgs,
                "tokens_total": tokens_total,
                "tokens_mean": tokens_total / len(scored),
                "per_category": per_cat,
                "elapsed_sec": time.time() - t0,
            },
            indent=2,
            ensure_ascii=False,
        ))
    print(f"\nWrote {len(rows)} rows to {out}", file=sys.stderr)
    print(f"Trajectories at {traj_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
