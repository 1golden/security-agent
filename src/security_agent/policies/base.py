"""Abstract policy interfaces.

Every Policy is a callable that maps (read-only) State to an Action.
Three flavors exist (Pre / Post / MemoryWrite) so trainers can attribute
rewards to each independently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from security_agent.types import Action, State


class Policy(ABC):
    """Base class for all policy types."""

    name: str = "policy"

    @abstractmethod
    def decide(self, state: State) -> Action:
        """Inspect state, emit one Action. Must NOT mutate state."""
        raise NotImplementedError


class PrePolicy(Policy):
    """Pre-retrieval planning: rewrite / decompose / noop."""

    name = "pre_policy"


class PostPolicy(Policy):
    """Post-retrieval verification: filter / retry / stop / answer."""

    name = "post_policy"


class MemoryWritePolicy(Policy):
    """Decides what (if anything) to write back after an answer is produced."""

    name = "memory_write_policy"
