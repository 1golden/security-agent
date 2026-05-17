"""Read JSONL trajectory logs back into Python objects.

Used by future RL training scripts to build behavior-cloning datasets and to
compute step-wise reward signals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_trajectories(path: str | Path) -> Iterator[dict]:
    """Yield each trajectory dict from a JSONL log file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def stage_actions(path: str | Path, stage: str) -> Iterator[tuple[dict, dict]]:
    """Yield (state_before, action) pairs for a given stage across all trajectories."""
    for traj in iter_trajectories(path):
        for step in traj.get("steps", []):
            if step.get("stage") == stage and step.get("action"):
                yield step["state_before"], step["action"]
