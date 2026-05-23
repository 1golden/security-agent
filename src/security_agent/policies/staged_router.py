"""Staged CQL-based router policies (RL-7).

Per-stage policies that consult a linear Q(s, a) function trained on v4
trajectory data (see ``rl/train_cql.py``), pick the argmax action type,
and **delegate** to whichever canonical policy (rule or llm) is the
majority producer of that action type in the training data.

Why delegation instead of emitting actions directly?
  CQL learned to predict action TYPES, not payload-bearing actions. A
  prediction of "REWRITE" needs an actual rewritten query string, which
  only the LLM PrePolicy knows how to produce. So we use the prediction
  as a *policy selector*: "this stage's action looks more like what
  Rule does — call Rule" or "looks more like LLM — call LLM".

Empirical action → producer mapping (from v4 replay data):

  pre  : noop / decompose                       → rule
         rewrite / choose_provider / inherit    → llm
  post : answer / narrow                        → rule
         gap_retry / broaden / filter           → llm
  write: write                                  → rule
         skip_write                             → llm
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from security_agent.policies.base import (
    MemoryWritePolicy,
    Policy,
    PostPolicy,
    PrePolicy,
)
from security_agent.policies.memory_write.llm import LLMWritePolicy
from security_agent.policies.memory_write.rule import RuleWritePolicy
from security_agent.policies.post_retrieval.llm import LLMPostPolicy
from security_agent.policies.post_retrieval.rule import RulePostPolicy
from security_agent.policies.pre_retrieval.llm import LLMPrePolicy
from security_agent.policies.pre_retrieval.rule import RulePrePolicy
from security_agent.types import Action, State


# Mirror the feature extractor used at training time.
# Keep this list in sync with rl/replay_dataset.py::FEATURE_KEYS.
FEATURE_KEYS = [
    "q_word_count", "q_short_factoid", "q_definition", "q_howto",
    "q_compound", "q_has_mitre_id", "q_has_cve", "q_has_owasp",
    "q_kw_mitigation", "q_kw_describe", "q_kw_session", "q_kw_logging",
    "retry_round", "n_docs_so_far", "coverage_so_far",
    "missing_count", "has_retrieved",
]

_MITRE_RE = re.compile(r"\b([TMGS]\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_OWASP_RE = re.compile(r"\bA0\d(?::20\d\d)?\b", re.IGNORECASE)
_HOW_RE = re.compile(r"^\s*how\b", re.IGNORECASE)
_DEF_RE = re.compile(r"^\s*what\s+(is|are|does|do)\b", re.IGNORECASE)
_CONJ_RE = re.compile(r"\b(and|then|also|as well as)\b", re.IGNORECASE)
_MITIGATION_KW = re.compile(r"\b(mitigat|prevent|defen[cs]|harden|protect)\b", re.IGNORECASE)
_DESCRIBE_KW = re.compile(r"\b(describ|explain|purpose of|role of)\b", re.IGNORECASE)
_SESSION_KW = re.compile(r"\b(session|cookie|jwt|authent|login)\b", re.IGNORECASE)
_LOGGING_KW = re.compile(r"\b(log\b|logging|audit|monitor|alert|detect)\b", re.IGNORECASE)


def _q_features(text: str) -> dict[str, float]:
    wc = len(re.findall(r"\S+", text))
    return {
        "q_word_count": min(wc, 40) / 40.0,
        "q_short_factoid": 1.0 if wc <= 6 else 0.0,
        "q_definition": 1.0 if _DEF_RE.match(text) else 0.0,
        "q_howto": 1.0 if _HOW_RE.match(text) else 0.0,
        "q_compound": 1.0 if (_CONJ_RE.search(text) and wc >= 6) else 0.0,
        "q_has_mitre_id": 1.0 if _MITRE_RE.search(text) else 0.0,
        "q_has_cve": 1.0 if _CVE_RE.search(text) else 0.0,
        "q_has_owasp": 1.0 if _OWASP_RE.search(text) else 0.0,
        "q_kw_mitigation": 1.0 if _MITIGATION_KW.search(text) else 0.0,
        "q_kw_describe": 1.0 if _DESCRIBE_KW.search(text) else 0.0,
        "q_kw_session": 1.0 if _SESSION_KW.search(text) else 0.0,
        "q_kw_logging": 1.0 if _LOGGING_KW.search(text) else 0.0,
    }


def _featurize(stage_state: dict) -> np.ndarray:
    return np.array(
        [float(stage_state.get(k, 0.0)) for k in FEATURE_KEYS],
        dtype=np.float32,
    )


# Empirical mapping: action.type → which base policy produces it most.
ACTION_TO_POLICY = {
    "pre": {
        "noop": "rule", "decompose": "rule",
        "rewrite": "llm", "choose_provider": "llm", "inherit_filter": "llm",
    },
    "post": {
        "answer": "rule", "narrow": "rule",
        "gap_retry": "llm", "broaden": "llm", "filter": "llm",
    },
    "write": {
        "write": "rule",
        "skip_write": "llm",
    },
}


@dataclass(slots=True)
class _StageModel:
    """Holds the linear Q matrix + action vocab for one stage."""

    W: np.ndarray                  # (n_actions, n_features + 1)
    actions: list[str]
    feature_keys: list[str]

    def predict_action(self, state_features: dict) -> str:
        x = _featurize(state_features)
        # prepend bias term
        xb = np.concatenate([[1.0], x])
        scores = self.W @ xb
        return self.actions[int(np.argmax(scores))]


_MODELS: dict[str, _StageModel] | None = None


def _load_models(path: str | Path) -> dict[str, _StageModel]:
    """Cache models the first time any policy is instantiated."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    raw = json.loads(Path(path).read_text())
    _MODELS = {
        stg: _StageModel(
            W=np.array(d["W"], dtype=np.float32),
            actions=list(d["actions"]),
            feature_keys=list(d["feature_keys"]),
        )
        for stg, d in raw.items()
    }
    return _MODELS


def _build_state_features(state: State, stage: str) -> dict[str, float]:
    """Construct the state-feature dict the CQL model was trained on.

    Pulls Q shape from state.question.text and dynamic signals from
    state.documents / state.coverage / state.attempt.
    """
    feats = _q_features(state.question.text)
    feats["retry_round"] = float(state.attempt)
    feats["n_docs_so_far"] = float(len(state.documents))
    cov = getattr(state, "coverage", None)
    feats["coverage_so_far"] = float(getattr(cov, "score", 0.0) or 0.0)
    feats["missing_count"] = float(len(getattr(cov, "missing_aspects", []) or []))
    feats["has_retrieved"] = 1.0 if state.documents else 0.0
    return feats


# ---------------------------------------------------------------------------
# Concrete per-stage policies
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "rl" / "data" / "cql_production_weights.json"


class CQLPrePolicy(PrePolicy):
    name = "pre.cql"

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        rule_arm: PrePolicy | None = None,
        llm_arm: PrePolicy | None = None,
    ) -> None:
        self._models = _load_models(weights_path)
        self._rule = rule_arm or RulePrePolicy()
        self._llm = llm_arm or LLMPrePolicy()

    def decide(self, state: State) -> Action:
        feats = _build_state_features(state, "pre")
        predicted = self._models["pre"].predict_action(feats)
        arm = ACTION_TO_POLICY["pre"].get(predicted, "rule")
        action = (self._llm if arm == "llm" else self._rule).decide(state)
        # Tag for trajectory analysis
        action.payload = dict(action.payload or {})
        action.payload.setdefault("_cql_predicted", predicted)
        action.payload.setdefault("_cql_arm", arm)
        return action


class CQLPostPolicy(PostPolicy):
    name = "post.cql"

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        rule_arm: PostPolicy | None = None,
        llm_arm: PostPolicy | None = None,
    ) -> None:
        self._models = _load_models(weights_path)
        self._rule = rule_arm or RulePostPolicy()
        self._llm = llm_arm or LLMPostPolicy()

    def decide(self, state: State) -> Action:
        feats = _build_state_features(state, "post")
        predicted = self._models["post"].predict_action(feats)
        arm = ACTION_TO_POLICY["post"].get(predicted, "rule")
        action = (self._llm if arm == "llm" else self._rule).decide(state)
        action.payload = dict(action.payload or {})
        action.payload.setdefault("_cql_predicted", predicted)
        action.payload.setdefault("_cql_arm", arm)
        return action


class CQLWritePolicy(MemoryWritePolicy):
    name = "write.cql"

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        rule_arm: MemoryWritePolicy | None = None,
        llm_arm: MemoryWritePolicy | None = None,
    ) -> None:
        self._models = _load_models(weights_path)
        self._rule = rule_arm or RuleWritePolicy()
        self._llm = llm_arm or LLMWritePolicy()

    def decide(self, state: State) -> Action:
        feats = _build_state_features(state, "write")
        predicted = self._models["write"].predict_action(feats)
        arm = ACTION_TO_POLICY["write"].get(predicted, "rule")
        action = (self._llm if arm == "llm" else self._rule).decide(state)
        action.payload = dict(action.payload or {})
        action.payload.setdefault("_cql_predicted", predicted)
        action.payload.setdefault("_cql_arm", arm)
        return action
