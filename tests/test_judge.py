"""Tests for the judge module."""
from __future__ import annotations

from security_agent.eval.judge import (
    DEFAULT_WEIGHTS,
    DIMENSIONS,
    HeuristicJudge,
    JudgeVerdict,
    LLMJudge,
    build_judge,
)
from security_agent.eval.score_trajectories import (
    score_jsonl_file,
    score_trajectory_dict,
    summarize,
)
from security_agent.generation.llm_generator import DummyLLM


# ---------- Heuristic ----------


def test_heuristic_empty_answer_scores_low():
    v = HeuristicJudge().score("what is X", "", evidence=[])
    assert v.score < 0.5
    assert v.dimensions["safety"] == 0.0  # empty answer is unsafe by our rule


def test_heuristic_groundedness_rewards_matching_citations():
    ev = [
        {"id": "doc1", "text": "SQL injection allows attackers to access data."},
        {"id": "doc2", "text": "Parameterized queries prevent it."},
    ]
    v = HeuristicJudge().score(
        "what is SQL injection",
        "SQL injection allows attackers to access data [doc1]. "
        "Parameterized queries prevent it [doc2].",
        evidence=ev,
    )
    assert v.dimensions["evidence_grounding"] == 1.0


def test_heuristic_groundedness_penalizes_unmatched_citations():
    ev = [{"id": "doc1", "text": "x"}]
    v = HeuristicJudge().score("q", "answer [doc99]", evidence=ev)
    # has citation but it's not in evidence → half credit
    assert 0.4 < v.dimensions["evidence_grounding"] < 0.6


def test_heuristic_safety_penalizes_exploit_markers():
    v = HeuristicJudge().score(
        "how to find vulns",
        "Step 1: scan target. Exploit payload: ' OR 1=1 --",
        evidence=[],
    )
    assert v.dimensions["safety"] < 0.5


def test_heuristic_factual_uses_ground_truth_when_present():
    ev: list[dict] = []
    a = "Sql injection happens when input is concatenated into queries"
    gt = "SQL injection happens when input is concatenated into queries"
    v = HeuristicJudge().score("what is sqli", a, evidence=ev, ground_truth=gt)
    assert v.dimensions["factual_accuracy"] > 0.6


def test_heuristic_aggregate_respects_weights():
    j = HeuristicJudge(weights={"safety": 1.0, "factual_accuracy": 0,
                                "evidence_grounding": 0, "completeness": 0})
    v = j.score("q", "harmless answer", evidence=[])
    # only safety counts → should be 1.0 (clean answer)
    assert v.score == 1.0


def test_judge_verdict_serializable():
    v = JudgeVerdict(score=0.7, dimensions={d: 0.7 for d in DIMENSIONS}, reasoning="r")
    d = v.to_dict()
    assert d["score"] == 0.7
    assert set(d["dimensions"]) == set(DIMENSIONS)


# ---------- LLM ----------


def test_llm_judge_with_dummy_backend_returns_neutral():
    # DummyLLM doesn't return valid JSON → judge falls back to {0.5,...}
    v = LLMJudge(llm=DummyLLM()).score("q", "a", evidence=[])
    assert 0.4 < v.score < 0.6


def test_build_judge_dispatch():
    assert isinstance(build_judge("heuristic"), HeuristicJudge)
    assert isinstance(build_judge("llm", llm=DummyLLM()), LLMJudge)


def test_default_weights_normalize_to_one():
    # weights stored internally must normalize even though defaults already sum to 1
    j = HeuristicJudge(weights={"factual_accuracy": 2.0})
    total = sum(j._weights.values())
    assert abs(total - 1.0) < 1e-6


# ---------- Trajectory scoring ----------


def test_score_trajectory_dict_basic():
    traj = {
        "question": {"text": "what is sqli", "session_id": "s"},
        "final_answer": "SQL injection is bad [d1]",
        "steps": [
            {"stage": "retrieve", "observation": {"doc_ids": ["d1"]}, "action": None,
             "step": 1, "state_before": {}},
        ],
    }
    v = score_trajectory_dict(traj, HeuristicJudge())
    assert v.score > 0


def test_summarize_handles_empty():
    assert summarize([])["n"] == 0


def test_summarize_aggregates_correctly():
    rows = [
        {"score": 0.6, "dimensions": {d: 0.6 for d in DIMENSIONS}, "blocked": False},
        {"score": 0.8, "dimensions": {d: 0.8 for d in DIMENSIONS}, "blocked": False},
    ]
    s = summarize(rows)
    assert s["n"] == 2
    assert abs(s["mean_score"] - 0.7) < 1e-9
    assert s["n_blocked"] == 0


def test_score_jsonl_file_end_to_end(tmp_path):
    p = tmp_path / "traj.jsonl"
    import json
    p.write_text(json.dumps({
        "question": {"text": "q"},
        "final_answer": "a",
        "steps": [],
        "metadata": {},
    }) + "\n", encoding="utf-8")
    rows = score_jsonl_file(str(p), HeuristicJudge())
    assert len(rows) == 1
    assert "score" in rows[0]
