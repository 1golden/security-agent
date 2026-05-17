"""Abstract memory store. Implementations: InMemory, A-Mem adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod

from security_agent.types import MemoryRecord


class MemoryStore(ABC):
    """Long-term memory. Records are written + retrieved by content / link / provider."""

    # ----- read paths -----

    @abstractmethod
    def recent(self, session_id: str, n: int) -> list[MemoryRecord]:
        """Cheap: return the last n records for this session, no embedding."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        session_id: str | None = None,
        providers: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Expensive: semantic search, optionally filtered."""

    @abstractmethod
    def neighbors(self, record_id: str, hops: int = 1) -> list[MemoryRecord]:
        """KG-style multi-hop. Follows the `links` field on each record."""

    @abstractmethod
    def preferred_providers(self, session_id: str | None = None, top_k: int = 3) -> list[str]:
        """Closed-loop feedback: which providers have helped before?"""

    # ----- write paths -----

    @abstractmethod
    def write(self, record: MemoryRecord) -> str:
        """Insert. Returns the assigned id."""

    @abstractmethod
    def update(self, record_id: str, patch: dict) -> None:
        """Mutate metadata or fields of an existing record."""

    @abstractmethod
    def link(self, src_id: str, dst_id: str) -> None:
        """Add an edge from src to dst."""

    @abstractmethod
    def record_provider_utility(
        self, session_id: str, provider: str, helped: bool
    ) -> None:
        """Feedback signal for preferred_providers()."""

    @abstractmethod
    def trigger_reflection(self, anchor_id: str) -> None:
        """A-Mem-style: re-examine neighbors of anchor and refine links/tags."""
