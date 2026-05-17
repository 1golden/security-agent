"""RL-trained post-retrieval policy. Interface stub."""
from __future__ import annotations

from security_agent.policies.base import PostPolicy
from security_agent.types import Action, ActionType, State


class RLPostPolicy(PostPolicy):
    name = "post.rl"

    def __init__(self, checkpoint: str | None = None) -> None:
        self._checkpoint = checkpoint
        self._ready = checkpoint is not None

    def decide(self, state: State) -> Action:
        if not self._ready:
            raise NotImplementedError(
                "RLPostPolicy needs a trained checkpoint."
            )
        return Action(type=ActionType.ANSWER, rationale="rl stub", confidence=0.0)
