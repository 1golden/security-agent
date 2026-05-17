"""Minimal CLI: `security-agent ask 'question'`."""
from __future__ import annotations

import argparse
import json
import sys

from security_agent.config import Config
from security_agent.eval_log.logger import TrajectoryLogger
from security_agent.pipeline import build_default_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="security-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="ask a single question")
    p_ask.add_argument("question")
    p_ask.add_argument("--session", default=None)
    p_ask.add_argument("--log-dir", default=None)
    p_ask.add_argument("--json", action="store_true", help="emit full trajectory as JSON")

    args = p.parse_args(argv)
    cfg = Config.from_env()
    log_dir = args.log_dir or cfg.pipeline.log_dir
    logger = TrajectoryLogger(log_dir=log_dir) if log_dir else None
    pipe = build_default_pipeline(cfg, logger=logger)

    if args.cmd == "ask":
        result = pipe.ask(args.question, session_id=args.session or "cli")
        if args.json:
            out = {
                "answer": result.answer,
                "blocked": result.blocked,
                "blocked_reason": result.blocked_reason,
                "trajectory": result.trajectory.to_dict(),
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            if result.blocked:
                print(f"[blocked] {result.blocked_reason}")
            else:
                print(result.answer or "(no answer)")
        if logger:
            logger.close()
            print(f"\n[log] trajectory at {logger.path}", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
