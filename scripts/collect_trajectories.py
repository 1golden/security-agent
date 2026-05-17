#!/usr/bin/env python3
"""Run a list of questions through the pipeline and append all trajectories.

The output JSONL is what you feed to behavior-cloning warm-start for the RL
policy. Two flags worth knowing:

  --policy rule|llm     which decision policy to record (default rule)
  --questions FILE      file with one question per line (default builtin set)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from security_agent.config import Config  # noqa: E402
from security_agent.eval_log.logger import TrajectoryLogger  # noqa: E402
from security_agent.pipeline import build_default_pipeline  # noqa: E402
from security_agent.types import Question  # noqa: E402


_BUILTIN_QUESTIONS = [
    "What is OWASP A03 injection?",
    "How do I mitigate broken access control per OWASP A01?",
    "What does NIST 800-53 AC family cover and how do I implement AC-2?",
    "Which technique does MITRE ATT&CK T1078 describe?",
    "Is CVE-2024-1234 a SQL injection?",
    "Explain MITRE ATT&CK T1078 and how to detect it",
    "And how do I prevent it",
    "What is in the CISA KEV catalog and how should I use it?",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["rule", "llm"], default="rule")
    p.add_argument("--out", default="runs/traj_collect.jsonl")
    p.add_argument("--questions", default=None)
    p.add_argument("--session", default="collect")
    args = p.parse_args()

    cfg = Config()
    cfg.pipeline.pre_policy = args.policy
    cfg.pipeline.post_policy = args.policy
    cfg.pipeline.write_policy = args.policy

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger = TrajectoryLogger(log_dir=str(out.parent), filename=out.name)
    pipe = build_default_pipeline(cfg, logger=logger)

    if args.questions:
        qs = [ln.strip() for ln in Path(args.questions).read_text().splitlines() if ln.strip()]
    else:
        qs = _BUILTIN_QUESTIONS

    for i, q in enumerate(qs):
        result = pipe.ask(Question(text=q, session_id=args.session))
        print(f"[{i+1:>3}/{len(qs)}] blocked={result.blocked} ans_len={len(result.answer or '')}")

    logger.close()
    print(f"\nWrote {len(qs)} trajectories to {logger.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
