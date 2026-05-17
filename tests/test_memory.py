"""InMemoryStore + probe + expansion."""
from __future__ import annotations

from security_agent.memory.expansion import MemoryExpansion
from security_agent.memory.inmem import InMemoryStore
from security_agent.memory.probe import FastMemoryProbe
from security_agent.types import MemoryRecord, Question, State


def _store_with_history() -> InMemoryStore:
    s = InMemoryStore()
    s.write(
        MemoryRecord(
            id="m1",
            content="OWASP A03 is about injection attacks.",
            keywords=["owasp", "a03", "injection"],
            metadata={"session_id": "sess-x", "provider": "owasp"},
        )
    )
    s.write(
        MemoryRecord(
            id="m2",
            content="Parameterize queries to prevent SQL injection.",
            keywords=["sql", "injection", "parameterize"],
            metadata={"session_id": "sess-x", "provider": "owasp"},
        )
    )
    s.link("m1", "m2")
    s.link("m2", "m1")
    s.record_provider_utility("sess-x", "owasp", helped=True)
    s.record_provider_utility("sess-x", "owasp", helped=True)
    return s


def test_recent_returns_latest_only():
    s = _store_with_history()
    recent = s.recent("sess-x", n=1)
    assert len(recent) == 1
    assert recent[0].id == "m2"


def test_search_keyword_overlap():
    s = _store_with_history()
    hits = s.search("sql injection", top_k=2)
    ids = {h.id for h in hits}
    assert "m2" in ids


def test_neighbors_follow_links():
    s = _store_with_history()
    nb = s.neighbors("m1", hops=1)
    assert any(r.id == "m2" for r in nb)


def test_preferred_providers_session_specific():
    s = _store_with_history()
    assert s.preferred_providers("sess-x", top_k=1) == ["owasp"]
    assert s.preferred_providers("sess-other", top_k=1) == []


def test_probe_extracts_coref_and_filters():
    s = _store_with_history()
    probe = FastMemoryProbe(s, recent_n=2)
    sig = probe(Question(text="How do I mitigate it in OWASP?", session_id="sess-x"))
    assert sig.recent_turns
    assert sig.explicit_filters.get("provider") == "owasp"
    # 'it' is in question; recent contains 'OWASP' / 'SQL' tokens
    assert "it" in sig.coref_hints or sig.coref_hints  # at least one resolved
    assert "owasp" in sig.preferred_providers


def test_expansion_combines_search_and_kg():
    s = _store_with_history()
    state = State(question=Question(text="injection mitigations", session_id="sess-x"))
    state.probe = FastMemoryProbe(s, recent_n=2)(state.question)
    expansion = MemoryExpansion(s, top_k=5, kg_hops=1)
    records = expansion(state)
    assert records, "expansion should surface at least one record"


def test_trigger_reflection_creates_symmetric_links():
    s = InMemoryStore()
    s.write(
        MemoryRecord(
            id="a", content="foo", keywords=["k1", "k2"], metadata={"session_id": "s"}
        )
    )
    s.write(
        MemoryRecord(
            id="b", content="bar", keywords=["k1", "k2"], metadata={"session_id": "s"}
        )
    )
    s.trigger_reflection("a")
    assert "b" in s._records["a"].links
    assert "a" in s._records["b"].links
