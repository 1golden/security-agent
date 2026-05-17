"""RL-trained memory-write policy. Interface stub."""
from __future__ import annotations

from security_agent.policies.base import MemoryWritePolicy
from security_agent.types import Action, ActionType, State


class RLWritePolicy(MemoryWritePolicy):
    name = "write.rl"

    def __init__(self, checkpoint: str | None = None) -> None:
        self._checkpoint = checkpoint
        self._ready = checkpoint is not None

    def decide(self, state: State) -> Action:
        if not self._ready:
            raise NotImplementedError("RLWritePolicy needs a trained checkpoint.")
        return Action(type=ActionType.SKIP_WRITE, rationale="rl stub", confidence=0.0)
