"""RL-trained pre-retrieval policy.

Interface stub: matches PrePolicy.decide() so the pipeline can swap it in
once a model is trained. Loading checkpoints, featurization, and inference
are deliberately left as the integration point.

Training data: read JSONL trajectory logs via security_agent.eval_log.replay
and filter stage == "pre" to get (state_before, action) supervision pairs.
"""
from __future__ import annotations

from security_agent.policies.base import PrePolicy
from security_agent.types import Action, ActionType, State


class RLPrePolicy(PrePolicy):
    name = "pre.rl"

    def __init__(self, checkpoint: str | None = None) -> None:
        self._checkpoint = checkpoint
        if checkpoint is None:
            self._ready = False
        else:
            # placeholder: real implementation loads a policy network here
            self._ready = True

    def decide(self, state: State) -> Action:
        if not self._ready:
            raise NotImplementedError(
                "RLPrePolicy has no checkpoint loaded. Train one with "
                "scripts/train_rl_policy.py, then pass checkpoint=<path>."
            )
        # placeholder action; replace with model.forward(state.features())
        return Action(type=ActionType.NOOP, rationale="rl stub", confidence=0.0)
