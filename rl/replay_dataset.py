#!/usr/bin/env python3
"""RL-2 — Replay dataset loader.

Reads the v4 .traj.jsonl files (rule, llm, router) and joins with the
v4 scored .jsonl files to attach per-trajectory rewards. Emits one row
per decision-bearing step (pre / post / write) with:

  - q_id          : question index (0..49)
  - policy        : "rule" | "llm" | "router"
  - stage         : "pre" | "post" | "write"
  - step_idx      : ordinal of this step within the trajectory
  - retry_round   : 0 for first pass, 1 for first retry, …
  - state         : dict of hand-crafted features (see featurize_state)
  - action        : str — the action.type taken at this step
  - reward        : float — final trajectory reward (terminal-only)

Writes:
  rl/data/replay.jsonl      one record per row
  rl/data/dataset_stats.txt one-page summary
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/mnt/d/Project/WORK/security-agent")
EXP = ROOT / "experiments" / "baseline_2026_05_19_v4"
OUT_DIR = ROOT / "rl" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICIES = ["rule", "llm", "router"]
DECISION_STAGES = {"pre", "post", "write"}

# Hand-crafted Q features. Mirror the (failed) E1 feature set so we can
# compare apples-to-apples whether step-level state buys us anything.
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


def q_features(q_text: str) -> dict[str, float]:
    low = q_text.lower()
    wc = len(re.findall(r"\S+", q_text))
    return {
        "q_word_count": min(wc, 40) / 40.0,
        "q_short_factoid": 1.0 if wc <= 6 else 0.0,
        "q_definition": 1.0 if _DEF_RE.match(q_text) else 0.0,
        "q_howto": 1.0 if _HOW_RE.match(q_text) else 0.0,
        "q_compound": 1.0 if (_CONJ_RE.search(q_text) and wc >= 6) else 0.0,
        "q_has_mitre_id": 1.0 if _MITRE_RE.search(q_text) else 0.0,
        "q_has_cve": 1.0 if _CVE_RE.search(q_text) else 0.0,
        "q_has_owasp": 1.0 if _OWASP_RE.search(q_text) else 0.0,
        "q_kw_mitigation": 1.0 if _MITIGATION_KW.search(q_text) else 0.0,
        "q_kw_describe": 1.0 if _DESCRIBE_KW.search(q_text) else 0.0,
        "q_kw_session": 1.0 if _SESSION_KW.search(q_text) else 0.0,
        "q_kw_logging": 1.0 if _LOGGING_KW.search(q_text) else 0.0,
    }


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def extract_rows_from_trajectory(traj: dict, policy: str, q_id: int,
                                  reward: float) -> list[dict]:
    """Walk a trajectory's steps and emit one row per decision-bearing step.

    The state attached to each row is the *causally-prior* observation —
    the model gets to see what the pipeline saw before it had to act.
    """
    q_text = safe_get(traj, "question", "text", default="")
    q_feats = q_features(q_text)

    rows: list[dict] = []
    last_coverage = None
    last_n_docs = None
    last_missing = None
    last_retrieve_obs = None
    retry_round = 0

    for step in traj.get("steps", []):
        stage = step.get("stage", "?")
        obs = step.get("observation") or {}
        action = step.get("action") or {}
        a_type = action.get("type") if isinstance(action, dict) else None

        # Track latest observations from non-decision stages so decision
        # stages can incorporate them into state.
        if stage == "retrieve":
            last_n_docs = obs.get("n_docs")
            last_retrieve_obs = obs
        elif stage == "coverage":
            last_coverage = obs.get("score")
            last_missing = len(obs.get("missing_aspects") or [])

        if stage not in DECISION_STAGES or a_type is None:
            continue

        # The pre-stage at retry_round > 0 sees the post-action that triggered the retry.
        state: dict = dict(q_feats)
        state["policy"] = policy
        state["stage"] = stage
        state["retry_round"] = float(retry_round)
        state["n_docs_so_far"] = float(last_n_docs or 0)
        state["coverage_so_far"] = float(last_coverage or 0.0)
        state["missing_count"] = float(last_missing or 0)
        # has_retrieved = saw a retrieve step yet?
        state["has_retrieved"] = 1.0 if last_retrieve_obs is not None else 0.0

        rows.append({
            "q_id": q_id,
            "policy": policy,
            "stage": stage,
            "step_idx": step.get("step", -1),
            "retry_round": retry_round,
            "state": state,
            "action": a_type,
            "reward": reward,
        })

        if stage == "post" and a_type == "gap_retry":
            retry_round += 1

    return rows


def load_v4_dataset() -> list[dict]:
    """Join trajectories with per-Q scores; emit step-level rows."""
    # Per-Q rewards (use patched router file since the unpatched .jsonl was
    # missing 4 Qs).
    rewards: dict[tuple[str, int], float] = {}
    for pol in POLICIES:
        scored_path = EXP / (
            f"bench50_router_v4_patched.jsonl" if pol == "router"
            else f"bench50_{pol}_v4.jsonl"
        )
        for line in open(scored_path):
            d = json.loads(line)
            s = d.get("score")
            if s is not None:
                rewards[(pol, d["i"])] = float(s)

    rows: list[dict] = []
    for pol in POLICIES:
        traj_path = EXP / f"bench50_{pol}_v4.traj.jsonl"
        for line in open(traj_path):
            traj = json.loads(line)
            sid = safe_get(traj, "question", "session_id", default="")
            # session_id format: "{pol}-v4-{NNN}"
            m = re.search(r"-(\d{3})$", sid or "")
            if not m:
                continue
            q_id = int(m.group(1))
            reward = rewards.get((pol, q_id))
            if reward is None:
                # router-v4 has 4 missing Qs (i=9,17,18,26) — skip them
                continue
            rows.extend(extract_rows_from_trajectory(traj, pol, q_id, reward))
    return rows


def main() -> None:
    rows = load_v4_dataset()
    out_path = OUT_DIR / "replay.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}")

    # Per-stage stats
    by_stage = defaultdict(list)
    by_stage_action = defaultdict(Counter)
    for r in rows:
        by_stage[r["stage"]].append(r)
        by_stage_action[r["stage"]][r["action"]] += 1

    stats_lines = [f"Total rows: {len(rows)}", ""]
    for stg in ("pre", "post", "write"):
        rs = by_stage[stg]
        stats_lines.append(f"=== {stg} (n={len(rs)}) ===")
        # action counts
        for a, c in by_stage_action[stg].most_common():
            stats_lines.append(f"  {a:18s}: {c}")
        # mean reward per action (which action led to higher reward on avg?)
        action_reward = defaultdict(list)
        for r in rs:
            action_reward[r["action"]].append(r["reward"])
        stats_lines.append("  mean reward by action:")
        for a in sorted(action_reward):
            v = action_reward[a]
            stats_lines.append(f"    {a:18s}: n={len(v):3d}  mean={sum(v)/len(v):.3f}")
        stats_lines.append("")

    stats_path = OUT_DIR / "dataset_stats.txt"
    stats_path.write_text("\n".join(stats_lines))
    print(f"wrote stats to {stats_path}")
    print()
    print("\n".join(stats_lines))


if __name__ == "__main__":
    main()
