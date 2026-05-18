"""Smoke tests for core data structures."""
from __future__ import annotations

from security_agent.types import (
    Action,
    ActionType,
    Budget,
    Coverage,
    Document,
    MemoryRecord,
    ProbeSignals,
    Question,
    State,
    Trajectory,
    TrajectoryStep,
)


def test_question_roundtrip():
    q = Question(text="What is OWASP A03?", user_id="u1")
    d = q.to_dict()
    assert d["text"] == "What is OWASP A03?"
    assert d["session_id"].startswith("sess-")


def test_state_effective_query_falls_back_to_question():
    s = State(question=Question(text="hello"))
    assert s.effective_query == "hello"
    s.rewritten_query = "hello world"
    assert s.effective_query == "hello world"


def test_state_coverage_delta_only_with_two_history_points():
    s = State(question=Question(text="x"))
    assert s.coverage_delta() is None
    s.coverage_history = [0.4]
    assert s.coverage_delta() is None
    s.coverage_history = [0.4, 0.55]
    assert abs(s.coverage_delta() - 0.15) < 1e-9


def test_budget_can_retry_respects_limits():
    """`target_attempt` is the attempt that WOULD run if we proceed."""
    b = Budget(max_retries=2, max_tokens=1000, min_coverage_delta=0.05)
    assert b.can_retry(1, None) is True   # 1st retry allowed
    assert b.can_retry(2, None) is True   # 2nd retry allowed
    assert b.can_retry(3, None) is False  # 3rd retry over budget
    b.tokens_spent = 1500
    assert b.can_retry(1, None) is False  # over token budget
    b.tokens_spent = 0
    assert b.can_retry(1, last_delta=0.01) is False  # delta too small
    assert b.can_retry(1, last_delta=0.10) is True


def test_action_type_round_trip():
    a = Action(type=ActionType.GAP_RETRY, payload={"aspect": "mitigation"})
    assert a.to_dict()["type"] == "gap_retry"


def test_coverage_default_is_empty_ok():
    c = Coverage()
    assert c.score == 0.0
    assert c.missing_aspects == []


def test_trajectory_append_and_serialize():
    t = Trajectory(question=Question(text="q"))
    t.append(
        TrajectoryStep(
            step=1, stage="pre", state_before={"x": 1}, action=None, observation={"ok": True}
        )
    )
    d = t.to_dict()
    assert len(d["steps"]) == 1
    assert d["steps"][0]["observation"]["ok"] is True


def test_memory_record_carries_ingestion_time():
    r = MemoryRecord(id="r1", content="hi")
    assert r.ingestion_time > 0


def test_probe_signals_serialize():
    p = ProbeSignals(recent_turns=["a"], kg_seed_nodes=["x"])
    d = p.to_dict()
    assert d["recent_turns"] == ["a"]


def test_document_to_dict():
    d = Document(id="d1", text="t", score=0.5, source="owasp")
    assert d.to_dict()["source"] == "owasp"
