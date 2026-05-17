"""Coverage check structural output."""
from __future__ import annotations

from security_agent.coverage import compute_coverage
from security_agent.types import Document, Question, State


def _state(query: str, docs: list[Document]) -> State:
    s = State(question=Question(text=query))
    s.documents = docs
    return s


def test_no_documents_yields_zero_score_and_missing_any_evidence():
    cov = compute_coverage(_state("what is X", []))
    assert cov.score == 0.0
    assert "any-evidence" in cov.missing_aspects or cov.missing_aspects == ["definition"]


def test_missing_mitigation_aspect_is_reported():
    docs = [Document(id="d1", text="injection is bad", score=0.5)]
    cov = compute_coverage(_state("how do I mitigate injection", docs))
    assert "mitigation" in cov.missing_aspects


def test_mitigation_aspect_satisfied_when_present():
    docs = [Document(id="d1", text="to mitigate injection use parameterized queries", score=0.7)]
    cov = compute_coverage(_state("how do I mitigate injection", docs))
    assert "mitigation" not in cov.missing_aspects


def test_redundancy_high_for_identical_docs():
    docs = [
        Document(id="a", text="parameterize queries to prevent sql injection", score=0.5),
        Document(id="b", text="parameterize queries to prevent sql injection", score=0.5),
    ]
    cov = compute_coverage(_state("prevent sql injection", docs))
    assert cov.redundancy > 0.8
