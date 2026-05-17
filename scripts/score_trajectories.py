#!/usr/bin/env python3
"""Score a trajectory JSONL with the LLM-as-judge (or heuristic).

Usage:
    python scripts/score_trajectories.py runs/traj.jsonl --judge heuristic
    python scripts/score_trajectories.py runs/traj.jsonl --judge llm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from security_agent.config import Config  # noqa: E402
from security_agent.eval.judge import build_judge  # noqa: E402
from security_agent.eval.score_trajectories import score_jsonl_file, summarize  # noqa: E402
from security_agent.generation.llm_generator import build_llm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="path to a trajectory JSONL")
    p.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--out", default=None, help="optional output JSONL with per-row scores")
    args = p.parse_args()

    cfg = Config()
    if args.judge == "llm":
        judge = build_judge("llm", llm=build_llm(cfg.llm))
    else:
        judge = build_judge("heuristic")

    rows = score_jsonl_file(args.jsonl, judge)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(rows)} rows to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
