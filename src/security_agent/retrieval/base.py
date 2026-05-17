"""Retriever interface. Implementations may also do rerank internally."""
from __future__ import annotations

from abc import ABC, abstractmethod

from security_agent.types import Document


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        providers: list[str] | None = None,
        expansion_terms: list[str] | None = None,
    ) -> list[Document]:
        """Run dense+sparse retrieval, rerank, and return top_k Documents."""
