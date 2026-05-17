"""JSONL trajectory logger.

One file per process; each line is one full Trajectory.to_dict(). This is the
primary data source for later RL training (behavior cloning warm-start + step-
wise reward attribution).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from security_agent.types import Trajectory


class TrajectoryLogger:
    def __init__(self, log_dir: str = "runs", filename: str | None = None) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        fname = filename or f"traj-{int(time.time())}-{os.getpid()}.jsonl"
        self._path = self._dir / fname
        self._fh = open(self._path, "a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, traj: Trajectory) -> None:
        self._fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
