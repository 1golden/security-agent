"""Tests for the router pre-policy."""
from __future__ import annotations

from security_agent.generation.llm_generator import DummyLLM
from security_agent.policies.pre_retrieval import build as build_pre
from security_agent.policies.pre_retrieval.router import (
    RouterPrePolicy,
    _features,
    choose,
)
from security_agent.types import ProbeSignals, Question, State


def test_router_choose_definition_goes_rule():
    assert choose("What is OWASP A03 injection?") == "rule"


def test_router_choose_short_factoid_goes_rule():
    assert choose("What is SQLi?") == "rule"


def test_router_choose_compound_goes_llm():
    assert choose("How do I mitigate SQL injection and how do I detect XSS") == "llm"


def test_router_choose_pronoun_open_howto_goes_llm():
    assert choose("How do I prevent it in a CI pipeline") == "llm"


def test_router_features_detect_signals():
    f = _features("What is OWASP A03?")
    assert f["is_definition"] is True
    assert f["is_short_factoid"] is True

    f2 = _features("how do I mitigate it and detect it")
    assert f2["has_compound"] is True
    assert f2["has_pronoun"] is True
    assert f2["is_open_howto"] is True


def test_router_picks_inner_policy_and_records_arm():
    r = RouterPrePolicy()
    state = State(question=Question(text="What is OWASP A03?"), probe=ProbeSignals())
    act = r.decide(state)
    assert r.last_arm == "rule"
    assert act.payload.get("_router_arm") == "rule"


def test_router_without_llm_falls_back_to_rule():
    """If the LLM arm wasn't wired but router picks LLM, fall back to rule."""
    r = RouterPrePolicy(llm_policy=None)
    state = State(
        question=Question(text="how do I prevent it in CI"),
        probe=ProbeSignals(),
    )
    act = r.decide(state)
    # router scored llm but fell back
    assert r.last_arm == "rule"
    assert act.payload.get("_router_arm") == "rule"


def test_build_dispatches_router():
    p = build_pre("router", llm=DummyLLM())
    assert isinstance(p, RouterPrePolicy)
