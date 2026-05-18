"""Tests for each policy family (Rule fully, LLM via DummyLLM, RL stub raises)."""
from __future__ import annotations

import pytest

from security_agent.generation.llm_generator import DummyLLM
from security_agent.policies.memory_write.llm import LLMWritePolicy
from security_agent.policies.memory_write.rl import RLWritePolicy
from security_agent.policies.memory_write.rule import RuleWritePolicy
from security_agent.policies.post_retrieval.llm import LLMPostPolicy
from security_agent.policies.post_retrieval.rl import RLPostPolicy
from security_agent.policies.post_retrieval.rule import RulePostPolicy
from security_agent.policies.pre_retrieval.llm import LLMPrePolicy
from security_agent.policies.pre_retrieval.rl import RLPrePolicy
from security_agent.policies.pre_retrieval.rule import RulePrePolicy
from security_agent.types import (
    ActionType,
    Budget,
    Coverage,
    Document,
    ProbeSignals,
    Question,
    State,
)


# ---------- Pre ----------


def test_rule_pre_inherits_filter_when_probe_has_explicit_one():
    state = State(
        question=Question(text="what is X in owasp"),
        probe=ProbeSignals(explicit_filters={"provider": "owasp"}),
    )
    act = RulePrePolicy().decide(state)
    assert act.type == ActionType.INHERIT_FILTER


def test_rule_pre_chooses_provider_from_preferred():
    state = State(
        question=Question(text="define injection"),
        probe=ProbeSignals(preferred_providers=["nist", "owasp"]),
    )
    act = RulePrePolicy().decide(state)
    assert act.type == ActionType.CHOOSE_PROVIDER
    assert "nist" in act.payload["providers"]


def test_rule_pre_rewrites_when_coref_hint_present():
    state = State(
        question=Question(text="how do I exploit it"),
        probe=ProbeSignals(coref_hints={"it": "OWASP_A03"}),
    )
    act = RulePrePolicy().decide(state)
    assert act.type == ActionType.REWRITE
    assert "OWASP_A03" in act.payload["query"]


def test_rule_pre_decomposes_conjunction():
    state = State(
        question=Question(text="what is OWASP A03 and how do I mitigate it in production"),
        probe=ProbeSignals(),
    )
    act = RulePrePolicy().decide(state)
    assert act.type == ActionType.DECOMPOSE
    assert len(act.payload["subqueries"]) >= 2


def test_rule_pre_noop_on_simple_query():
    state = State(question=Question(text="hello"), probe=ProbeSignals())
    assert RulePrePolicy().decide(state).type == ActionType.NOOP


def test_llm_pre_returns_valid_action_with_dummy_backend():
    state = State(question=Question(text="anything"), probe=ProbeSignals())
    act = LLMPrePolicy(llm=DummyLLM()).decide(state)
    # DummyLLM returns invalid JSON → falls back to NOOP
    assert act.type == ActionType.NOOP


def test_rl_pre_without_checkpoint_raises():
    with pytest.raises(NotImplementedError):
        RLPrePolicy().decide(State(question=Question(text="x")))


# ---------- Post ----------


def test_rule_post_answers_when_coverage_good():
    s = State(question=Question(text="q"))
    s.coverage = Coverage(score=0.9, missing_aspects=[])
    s.documents = [Document(id="d1", text="t", score=0.5)]
    s.budget = Budget()
    assert RulePostPolicy(threshold=0.5).decide(s).type == ActionType.ANSWER


def test_rule_post_gap_retries_specific_aspect():
    s = State(question=Question(text="q"))
    s.coverage = Coverage(score=0.3, missing_aspects=["mitigation"])
    s.budget = Budget(max_retries=2)
    act = RulePostPolicy(threshold=0.5).decide(s)
    assert act.type == ActionType.GAP_RETRY
    assert act.payload["aspect"] == "mitigation"


def test_rule_post_broadens_when_low_score_no_aspect():
    s = State(question=Question(text="q"))
    s.coverage = Coverage(score=0.2, missing_aspects=[])
    s.budget = Budget(max_retries=2)
    s.documents = [Document(id="d", text="t", score=0.1)]
    assert RulePostPolicy(threshold=0.5).decide(s).type == ActionType.BROADEN


def test_rule_post_narrows_when_redundant():
    s = State(question=Question(text="q"))
    s.coverage = Coverage(score=0.4, missing_aspects=[], redundancy=0.9)
    s.budget = Budget(max_retries=2)
    s.documents = [Document(id="d", text="t", score=0.5)]
    assert RulePostPolicy(threshold=0.5).decide(s).type == ActionType.NARROW


def test_rule_post_stops_when_budget_exhausted_and_no_docs():
    """With max_retries=0, post.decide must NOT recommend any retry action;
    it falls through to STOP when no documents exist."""
    s = State(question=Question(text="q"))
    s.coverage = Coverage(score=0.2)
    s.budget = Budget(max_retries=0)
    assert RulePostPolicy().decide(s).type == ActionType.STOP


def test_rl_post_without_checkpoint_raises():
    with pytest.raises(NotImplementedError):
        RLPostPolicy().decide(State(question=Question(text="x")))


# ---------- Write ----------


def test_rule_write_skips_when_no_answer():
    s = State(question=Question(text="q"))
    s.finished = False
    s.answer = None
    assert RuleWritePolicy().decide(s).type == ActionType.SKIP_WRITE


def test_rule_write_writes_facts_when_coverage_decent():
    s = State(question=Question(text="q"))
    s.finished = True
    s.answer = "Use parameterized queries to prevent SQLi."
    s.coverage = Coverage(score=0.7)
    s.documents = [Document(id="d", text="x")]
    act = RuleWritePolicy().decide(s)
    assert act.type == ActionType.WRITE
    assert act.payload["axes"]["facts"] is True


def test_rule_write_marks_constraints_on_imperative_answer():
    s = State(question=Question(text="q"))
    s.finished = True
    s.answer = "You must enable MFA and never store plaintext passwords."
    s.coverage = Coverage(score=0.7)
    s.documents = [Document(id="d", text="x")]
    act = RuleWritePolicy().decide(s)
    assert act.type == ActionType.WRITE
    assert act.payload["axes"]["constraints"] is True


def test_rule_write_keeps_aliases_off_when_no_coref():
    s = State(question=Question(text="q"))
    s.finished = True
    s.answer = "OWASP A03 is about injection."
    s.coverage = Coverage(score=0.8)
    s.documents = [Document(id="d", text="x")]
    act = RuleWritePolicy().decide(s)
    assert act.payload["axes"]["aliases"] is False


def test_llm_write_skips_on_no_answer():
    s = State(question=Question(text="q"))
    assert LLMWritePolicy(llm=DummyLLM()).decide(s).type == ActionType.SKIP_WRITE


def test_rl_write_without_checkpoint_raises():
    with pytest.raises(NotImplementedError):
        RLWritePolicy().decide(State(question=Question(text="x")))
