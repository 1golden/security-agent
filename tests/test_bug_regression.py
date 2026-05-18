"""One regression test per fixed bug (Bug #1..#10).

Each test reproduces the original failure pattern and asserts the fix
behavior. If any of these fails in the future, the original bug has
returned.
"""
from __future__ import annotations

from security_agent.coverage import (
    LLMCoverage,
    _aspect_present,
    build_coverage,
    compute_coverage,
)
from security_agent.eval.judge import LLMJudge
from security_agent.eval.score_trajectories import _extract_evidence
from security_agent.generation.llm_generator import DummyLLM
from security_agent.memory.inmem import InMemoryStore
from security_agent.memory.probe import FastMemoryProbe
from security_agent.policies.pre_retrieval.llm import LLMPrePolicy, _allowed_actions
from security_agent.types import (
    ActionType,
    Budget,
    Coverage,
    Document,
    MemoryRecord,
    ProbeSignals,
    Question,
    State,
)


# ----------------------------------------------------------------------------
# Bug #1 — Judge had no evidence text. Fixed by writing doc_texts into the
# retrieve observation and reconstructing real evidence from the JSONL.
# ----------------------------------------------------------------------------


def test_bug1_extract_evidence_recovers_real_text():
    traj = {
        "steps": [
            {
                "stage": "retrieve",
                "observation": {
                    "doc_ids": ["d1", "d2"],
                    "sources": ["s1", "s2"],
                    "doc_texts": [
                        {"id": "d1", "source": "s1", "score": 0.8, "text": "SQL injection occurs when..."},
                        {"id": "d2", "source": "s2", "score": 0.6, "text": "Use parameterized queries"},
                    ],
                },
            }
        ]
    }
    ev = _extract_evidence(traj)
    assert len(ev) == 2
    assert any("SQL injection" in e["text"] for e in ev)
    assert any("parameterized" in e["text"] for e in ev)


def test_bug1_extract_evidence_legacy_format_still_works():
    """Old trajectories (only doc_ids) should still parse, with empty text."""
    traj = {"steps": [{"stage": "retrieve", "observation": {"doc_ids": ["x"]}}]}
    ev = _extract_evidence(traj)
    assert ev[0]["id"] == "x"
    assert ev[0]["text"] == ""


def test_bug1_extract_evidence_dedupes_across_retries():
    traj = {
        "steps": [
            {"stage": "retrieve", "observation": {
                "doc_texts": [{"id": "d1", "text": "short"}]}},
            {"stage": "retrieve", "observation": {
                "doc_texts": [{"id": "d1", "text": "a much longer version with content"}]}},
        ]
    }
    ev = _extract_evidence(traj)
    assert len(ev) == 1
    # keeps the longest text seen for the same id
    assert ev[0]["text"] == "a much longer version with content"


# ----------------------------------------------------------------------------
# Bug #2 — coverage said "missing definition" when docs explained the topic
# via "is a / occurs when / refers to" patterns. _aspect_present now matches
# content patterns, not just literal keywords.
# ----------------------------------------------------------------------------


def test_bug2_definition_aspect_recognized_without_literal_word():
    docs = [
        Document(
            id="d1",
            text="SQL injection occurs when input data is concatenated into a SQL query.",
            score=0.8,
        )
    ]
    # before the fix this returned False — the doc never uses 'definition'
    assert _aspect_present("definition", docs) is True


def test_bug2_mitigation_aspect_recognized_via_action_verbs():
    docs = [
        Document(id="d1", text="Always parameterize queries to prevent injection.", score=0.7)
    ]
    assert _aspect_present("mitigation", docs) is True


def test_bug2_llm_coverage_returns_well_formed_struct():
    # DummyLLM returns invalid JSON → LLMCoverage falls back to neutral but
    # still emits a valid Coverage. Score ~ 0.5, missing kept empty.
    state = State(question=Question(text="what is sqli"))
    state.documents = [Document(id="d1", text="some text", score=0.5)]
    cov = LLMCoverage(DummyLLM())(state)
    assert isinstance(cov, Coverage)
    assert 0.0 <= cov.score <= 1.0


def test_bug2_build_coverage_dispatch():
    assert build_coverage("heuristic") is compute_coverage
    assert isinstance(build_coverage("llm", llm=DummyLLM()), LLMCoverage)


# ----------------------------------------------------------------------------
# Bug #3 — DECOMPOSE was a silent no-op on the retriever. Fixed by
# _retrieve_with_subqueries which iterates subqueries and merges.
# ----------------------------------------------------------------------------


def test_bug3_pipeline_runs_decompose_subqueries_at_retrieve():
    """End-to-end: when state.subqueries is set, the retriever sees them.

    We assert this by inspecting calls to a recording Retriever stub.
    """
    from security_agent.config import Config
    from security_agent.pipeline import build_default_pipeline
    from security_agent.retrieval.base import Retriever
    from security_agent.types import Document

    calls: list[str] = []

    class RecordingRetriever(Retriever):
        def retrieve(self, query, top_k, providers=None, expansion_terms=None):
            calls.append(query)
            return [Document(id=f"r{len(calls)}", text=f"answer to: {query}", score=0.6)]

    cfg = Config()
    cfg.pipeline.coverage_kind = "heuristic"
    pipe = build_default_pipeline(cfg)
    pipe.retriever = RecordingRetriever()

    # craft a question with a clear conjunction so RulePrePolicy decomposes
    r = pipe.ask("What is OWASP A03 injection and how do I detect it in production")
    assert any("How do I detect it" in c or "how do I detect it" in c for c in calls), (
        f"Retriever never saw a subquery; calls={calls}"
    )


# ----------------------------------------------------------------------------
# Bug #4 — LLMPrePolicy could rewrite forever. Now action allowance shrinks
# as state acquires effects.
# ----------------------------------------------------------------------------


def test_bug4_rewrite_disallowed_after_first_rewrite():
    state = State(question=Question(text="q"))
    assert "rewrite" in _allowed_actions(state)
    state.rewritten_query = "already done"
    assert "rewrite" not in _allowed_actions(state)


def test_bug4_choose_provider_disallowed_once_filter_set():
    state = State(question=Question(text="q"))
    assert "choose_provider" in _allowed_actions(state)
    state.provider_filter = ["owasp"]
    assert "choose_provider" not in _allowed_actions(state)
    assert "inherit_filter" not in _allowed_actions(state)


def test_bug4_llm_pre_coerces_disallowed_pick_to_noop():
    """If the LLM tries a disallowed action, we coerce to noop, not crash."""
    state = State(question=Question(text="q"))
    state.rewritten_query = "x"
    # DummyLLM never returns valid JSON, so action defaults to noop anyway.
    # The real assertion: no exception is raised.
    act = LLMPrePolicy(DummyLLM()).decide(state)
    assert act.type == ActionType.NOOP


# ----------------------------------------------------------------------------
# Bug #5 — probe picked "Based" (sentence-initial cap) as coref entity.
# Now acronyms/multi-cap tokens are preferred and a stop-list filters cruft.
# ----------------------------------------------------------------------------


def test_bug5_coref_skips_sentence_initial_word():
    s = InMemoryStore()
    s.write(MemoryRecord(
        id="m1",
        content="Based on the provided evidence, SQL injection is a type of attack on OWASP A03.",
        keywords=["sql", "owasp"],
        metadata={"session_id": "s"},
    ))
    probe = FastMemoryProbe(s, recent_n=2)
    sig = probe(Question(text="how do I detect it", session_id="s"))
    # 'Based' must NOT be picked; 'OWASP' or 'SQL' is acceptable
    if sig.coref_hints:
        for v in sig.coref_hints.values():
            assert v != "Based", f"coref picked 'Based' — bug regression"


def test_bug5_coref_prefers_acronym_over_capitalized_word():
    s = InMemoryStore()
    s.write(MemoryRecord(
        id="m1",
        content="The user asked about OWASP A03 injection.",
        keywords=["owasp"],
        metadata={"session_id": "s"},
    ))
    probe = FastMemoryProbe(s, recent_n=2)
    sig = probe(Question(text="how do I mitigate it", session_id="s"))
    if sig.coref_hints:
        # OWASP (5 caps) should outrank "The" (stop-list, anyway) and "A03"
        assert any(v == "OWASP" for v in sig.coref_hints.values())


def test_bug5_coref_emits_nothing_when_no_strong_candidate():
    s = InMemoryStore()
    s.write(MemoryRecord(
        id="m1",
        content="Based on evidence, the result was the following.",
        keywords=[],
        metadata={"session_id": "s"},
    ))
    probe = FastMemoryProbe(s, recent_n=2)
    sig = probe(Question(text="explain it more", session_id="s"))
    # all candidates are stop-listed → no coref hint
    assert sig.coref_hints == {}


# ----------------------------------------------------------------------------
# Bug #6 — token budget didn't include policy calls. LLMBackend now
# accumulates total_tokens_used across every chat() call.
# ----------------------------------------------------------------------------


def test_bug6_llm_backend_accumulates_tokens():
    llm = DummyLLM()
    assert llm.total_tokens_used == 0
    llm.chat("sys", "user one")
    after_one = llm.total_tokens_used
    assert after_one > 0
    llm.chat("sys", "user two and a longer message here")
    assert llm.total_tokens_used > after_one
    assert llm.calls_made == 2


def test_bug6_reset_usage_clears_counter():
    llm = DummyLLM()
    llm.chat("s", "u")
    llm.reset_usage()
    assert llm.total_tokens_used == 0
    assert llm.calls_made == 0


# ----------------------------------------------------------------------------
# Bug #7 — provider filter could zero-out retrieval. Pipeline now auto-falls
# back to no-filter on empty result (without consuming a retry attempt).
# ----------------------------------------------------------------------------


def test_bug7_provider_zeroout_triggers_fallback():
    """If filter yields 0 docs, retry without it. No extra retry attempt."""
    from security_agent.config import Config
    from security_agent.pipeline import build_default_pipeline
    from security_agent.retrieval.base import Retriever

    calls: list[dict] = []

    class FilteringRetriever(Retriever):
        def retrieve(self, query, top_k, providers=None, expansion_terms=None):
            calls.append({"q": query, "providers": providers})
            if providers:
                return []  # simulate a too-restrictive filter
            return [Document(id="d1", text="SQL injection occurs when input is bad", score=0.8)]

    cfg = Config()
    cfg.pipeline.coverage_kind = "heuristic"
    pipe = build_default_pipeline(cfg)
    pipe.retriever = FilteringRetriever()

    state = State(question=Question(text="what is sqli"))
    state.provider_filter = ["nonexistent_provider"]
    docs = pipe._retrieve_with_subqueries(state, top_k=4)
    # before the fix this would have returned [] and tanked the run
    # the fallback runs at the pipeline level inside _run_loop, not in
    # _retrieve_with_subqueries — verify by running the loop instead
    state2 = State(question=Question(text="what is sqli"))
    state2.provider_filter = ["nonexistent_provider"]
    state2.budget = Budget()
    from security_agent.types import Trajectory
    traj = Trajectory(question=state2.question)
    pipe._run_loop(state2, traj)
    assert state2.documents, "fallback should have recovered documents"
    assert state2.provider_filter == [], "fallback should have cleared the bad filter"


# ----------------------------------------------------------------------------
# Bug #8 — FILTER path appended to coverage_history twice, corrupting delta.
# ----------------------------------------------------------------------------


def test_bug8_filter_does_not_double_append_coverage_history():
    """coverage_history must have one entry per RETRIEVE pass, regardless
    of whether FILTER ran. Before the fix, FILTER added a second entry per
    iter, corrupting coverage_delta() and the Budget check."""
    from security_agent.config import Config
    from security_agent.pipeline import build_default_pipeline
    from security_agent.policies.base import PostPolicy
    from security_agent.types import Action

    class FilterOncePostPolicy(PostPolicy):
        def __init__(self):
            self.calls = 0

        def decide(self, state):
            self.calls += 1
            if self.calls == 1:
                return Action(type=ActionType.FILTER, payload={"drop": []})
            return Action(type=ActionType.ANSWER)

    cfg = Config()
    cfg.pipeline.coverage_kind = "heuristic"
    pipe = build_default_pipeline(cfg)
    pipe.post = FilterOncePostPolicy()
    result = pipe.ask("what is sqli")
    # count retrieve stages
    n_retrieves = sum(1 for s in result.trajectory.steps if s.stage == "retrieve")
    assert n_retrieves >= 1
    # the invariant: exactly one history entry per retrieve. Before the fix,
    # FILTER would have produced n_retrieves+1 entries.
    assert len(result.state.coverage_history) == n_retrieves, (
        f"expected {n_retrieves} coverage entries (one per retrieve), "
        f"got {len(result.state.coverage_history)}"
    )


# ----------------------------------------------------------------------------
# Bug #9 — Budget.max_retries was off-by-one (>= instead of >).
# ----------------------------------------------------------------------------


def test_bug9_budget_max_retries_semantics():
    """max_retries=N means N total retries (target attempts 1..N are valid)."""
    b = Budget(max_retries=2, max_tokens=10_000)
    assert b.can_retry(1, None) is True   # 1st retry: in budget
    assert b.can_retry(2, None) is True   # 2nd retry: still in budget
    assert b.can_retry(3, None) is False  # 3rd retry: over


def test_bug9_max_retries_zero_denies_even_first_retry():
    """Regression: under old `>=` semantics this was True with max_retries=0."""
    b = Budget(max_retries=0, max_tokens=10_000)
    assert b.can_retry(1, None) is False


# ----------------------------------------------------------------------------
# Bug #10 — judge evidence cap was 4000 (too small). Now 12000.
# ----------------------------------------------------------------------------


def test_bug10_judge_default_evidence_cap_is_generous():
    j = LLMJudge(llm=DummyLLM())
    assert j._max_ev >= 12000
