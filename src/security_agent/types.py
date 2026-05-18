"""Core data structures shared across the pipeline.

These are intentionally small and easy to serialize so they flow cleanly into
the trajectory JSONL log and can later feed RL training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class ActionType(str, Enum):
    # --- pre-retrieval (planning) ---
    NOOP = "noop"
    REWRITE = "rewrite"
    DECOMPOSE = "decompose"
    INHERIT_FILTER = "inherit_filter"        # reuse a filter from prior turn / probe
    CHOOSE_PROVIDER = "choose_provider"      # bias retrieval toward a source

    # --- post-retrieval (verification) ---
    ANSWER = "answer"
    FILTER = "filter"
    BROADEN = "broaden"                      # retry with looser scope
    NARROW = "narrow"                        # retry with tighter scope
    GAP_RETRY = "gap_retry"                  # retry targeting a specific missing aspect
    STOP = "stop"

    # --- memory write (multi-axis, see Action.payload) ---
    WRITE = "write"                          # generic write; payload describes axes
    SKIP_WRITE = "skip_write"


# Sub-axes of a WRITE action (each independently on/off in the payload).
WRITE_AXIS_FACTS = "facts"                  # raw answer-grounded facts (safe)
WRITE_AXIS_CONSTRAINTS = "constraints"      # user preferences / hard rules
WRITE_AXIS_ALIASES = "aliases"              # entity / coreference mappings (poison risk)
WRITE_AXIS_REFLECTION = "reflection"        # trigger A-Mem-style evolution (expensive)
WRITE_AXIS_PROVIDER_UTILITY = "provider_utility"  # which provider helped (feedback loop)

ALL_WRITE_AXES = (
    WRITE_AXIS_FACTS,
    WRITE_AXIS_CONSTRAINTS,
    WRITE_AXIS_ALIASES,
    WRITE_AXIS_REFLECTION,
    WRITE_AXIS_PROVIDER_UTILITY,
)


@dataclass
class Budget:
    """Hard limits enforced by the post-retrieval loop."""

    max_retries: int = 2
    max_tokens: int = 8000              # cumulative LLM tokens this episode may spend
    min_coverage_delta: float = 0.05    # if a retry doesn't improve coverage by this, stop
    tokens_spent: int = 0

    def can_retry(self, target_attempt: int, last_delta: float | None) -> bool:
        """Bug #9 fix — clean semantics:

        ``target_attempt`` is the attempt number that would be EXECUTED if we
        proceed. E.g. ``target_attempt=1`` means "we are about to start the
        first retry (attempt #1)". This way:

        - post.decide passes ``state.attempt + 1`` (the retry it would
          recommend)
        - the loop, after bumping ``state.attempt``, passes ``state.attempt``
          directly (it's already the post-bump target)

        Both callsites ask the same question: is attempt #target within
        budget?

        ``max_retries=N`` means up to N retries are allowed, i.e. valid
        ``target_attempt`` values are 1..N. Anything beyond is denied.
        ``max_retries=0`` correctly denies even attempt #1 — so post.decide
        falls through to ANSWER/STOP without recommending a doomed retry.
        """
        if target_attempt > self.max_retries:
            return False
        if self.tokens_spent >= self.max_tokens:
            return False
        if last_delta is not None and last_delta < self.min_coverage_delta:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "min_coverage_delta": self.min_coverage_delta,
            "tokens_spent": self.tokens_spent,
        }


@dataclass
class Question:
    """User-facing request. `session_id` ties multiple turns together."""

    text: str
    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:8]}")
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """A single retrieved chunk."""

    id: str
    text: str
    score: float = 0.0
    source: str | None = None              # provider name (owasp / nist / cisa / ...)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class MemoryRecord:
    """An item read from (or to be written to) the long-term memory."""

    id: str
    content: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    context: str = ""
    links: list[str] = field(default_factory=list)
    event_time: float | None = None                            # when the event happened
    ingestion_time: float = field(default_factory=time.time)   # when it was written
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "keywords": self.keywords,
            "tags": self.tags,
            "context": self.context,
            "links": self.links,
            "event_time": self.event_time,
            "ingestion_time": self.ingestion_time,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class ProbeSignals:
    """Cheap signals produced by Fast Memory Probe.

    Designed to be lightweight — no embedding calls, no LLM. Just lookups
    against indexed session state and a small KG seed search.
    """

    recent_turns: list[str] = field(default_factory=list)
    explicit_filters: dict[str, Any] = field(default_factory=dict)  # e.g. {"provider": "nist"}
    coref_hints: dict[str, str] = field(default_factory=dict)       # pronoun → entity
    kg_seed_nodes: list[str] = field(default_factory=list)
    preferred_providers: list[str] = field(default_factory=list)    # learned from past writes

    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_turns": list(self.recent_turns),
            "explicit_filters": dict(self.explicit_filters),
            "coref_hints": dict(self.coref_hints),
            "kg_seed_nodes": list(self.kg_seed_nodes),
            "preferred_providers": list(self.preferred_providers),
        }


@dataclass
class Coverage:
    """Structured output of the coverage check.

    A single float is not enough — the post-retrieval policy needs to choose
    between gap-retry / broaden / narrow, and that requires knowing *what*
    is missing or low-confidence, not just *how much*.
    """

    score: float = 0.0
    missing_aspects: list[str] = field(default_factory=list)
    low_confidence_claims: list[str] = field(default_factory=list)
    redundancy: float = 0.0                                    # 0..1, high = too narrow
    diagnosis: str = ""                                        # short human-readable summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "missing_aspects": list(self.missing_aspects),
            "low_confidence_claims": list(self.low_confidence_claims),
            "redundancy": self.redundancy,
            "diagnosis": self.diagnosis,
        }


@dataclass
class Action:
    """A decision emitted by a Policy."""

    type: ActionType
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "payload": self.payload,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class State:
    """Mutable working state threaded through the pipeline.

    The pipeline keeps one State per Question. Policies inspect it (treat as
    read-only) and return Actions; the pipeline mutates State based on them.
    Pure-function policies make testing and later RL training tractable.
    """

    question: Question
    probe: ProbeSignals = field(default_factory=ProbeSignals)
    rewritten_query: str | None = None
    subqueries: list[str] = field(default_factory=list)
    provider_filter: list[str] = field(default_factory=list)   # selected by CHOOSE_PROVIDER
    inherited_filters: dict[str, Any] = field(default_factory=dict)
    expansion_terms: list[str] = field(default_factory=list)
    memory_records: list[MemoryRecord] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    coverage_history: list[float] = field(default_factory=list)  # to compute delta over retries
    answer: str | None = None
    attempt: int = 0
    budget: Budget = field(default_factory=Budget)
    finished: bool = False

    @property
    def effective_query(self) -> str:
        return self.rewritten_query or self.question.text

    def coverage_delta(self) -> float | None:
        """How much did the latest coverage improve over the previous one?"""
        if len(self.coverage_history) < 2:
            return None
        return self.coverage_history[-1] - self.coverage_history[-2]

    def snapshot(self) -> dict[str, Any]:
        """Lightweight serialization for the trajectory log."""
        return {
            "rewritten_query": self.rewritten_query,
            "subqueries": list(self.subqueries),
            "provider_filter": list(self.provider_filter),
            "expansion_terms": list(self.expansion_terms),
            "n_probe_signals": len(self.probe.recent_turns)
            + len(self.probe.kg_seed_nodes),
            "n_memory_records": len(self.memory_records),
            "n_documents": len(self.documents),
            "coverage": self.coverage.to_dict(),
            "coverage_delta": self.coverage_delta(),
            "attempt": self.attempt,
            "budget": self.budget.to_dict(),
            "answer_len": len(self.answer) if self.answer else 0,
            "finished": self.finished,
        }


@dataclass
class TrajectoryStep:
    """One (state, action, observation) tuple. The unit of RL data."""

    step: int
    stage: str  # input_safety | probe | pre | expansion | retrieve | coverage |
                # post | generate | write | output_safety
    state_before: dict[str, Any]
    action: dict[str, Any] | None
    observation: dict[str, Any] | None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "stage": self.stage,
            "state_before": self.state_before,
            "action": self.action,
            "observation": self.observation,
            "timestamp": self.timestamp,
        }


@dataclass
class Trajectory:
    """The full trace of one Question → Answer episode."""

    question: Question
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str | None = None
    reward: float | None = None                                # filled by evaluator
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "reward": self.reward,
            "metadata": self.metadata,
        }
