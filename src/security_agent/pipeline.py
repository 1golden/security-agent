"""Main pipeline orchestrator.

Stages (in order):
  0. Input Safety
  1. Fast Memory Probe
  2. Pre-Retrieval Policy        (loops here with Post)
  3. Memory Expansion
  4. RAG Retrieval + Rerank      (now multi-query if DECOMPOSE was chosen)
  5. Coverage Check              (heuristic or LLM; pluggable)
  6. Post-Retrieval Policy        — may emit retry (broaden/narrow/gap_retry)
  7. Answer Generation
  8. Memory Write Policy + Memory Write
  9. Output Safety

Each stage emits a TrajectoryStep with (state_before, action, observation).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

from security_agent.config import Config
from security_agent.coverage import build_coverage, compute_coverage
from security_agent.eval_log.logger import TrajectoryLogger
from security_agent.generation.llm_generator import LLMBackend, LLMGenerator, build_llm
from security_agent.memory.base import MemoryStore
from security_agent.memory.expansion import MemoryExpansion
from security_agent.memory.inmem import InMemoryStore
from security_agent.memory.probe import FastMemoryProbe
from security_agent.policies import (
    memory_write as write_pkg,
    post_retrieval as post_pkg,
    pre_retrieval as pre_pkg,
)
from security_agent.policies.base import MemoryWritePolicy, PostPolicy, PrePolicy
from security_agent.retrieval.base import Retriever
from security_agent.retrieval.dummy import DummyRetriever
from security_agent.safety.wrappers import InputSafety, OutputSafety
from security_agent.types import (
    Action,
    ActionType,
    Budget,
    Coverage,
    Document,
    MemoryRecord,
    Question,
    State,
    Trajectory,
    TrajectoryStep,
    WRITE_AXIS_ALIASES,
    WRITE_AXIS_FACTS,
    WRITE_AXIS_PROVIDER_UTILITY,
    WRITE_AXIS_REFLECTION,
)


# Cap per-doc text size in the trajectory observation. Big enough that the
# judge can verify factual claims; small enough that the JSONL log stays
# manageable. Bug #1 fix.
_OBS_DOC_TEXT_CAP = 800


@dataclass
class PipelineResult:
    answer: str | None
    trajectory: Trajectory
    state: State
    blocked: bool = False
    blocked_reason: str = ""


class Pipeline:
    """End-to-end orchestrator. Treat as the single public entry point."""

    def __init__(
        self,
        memory: MemoryStore,
        retriever: Retriever,
        generator: LLMGenerator,
        pre_policy: PrePolicy,
        post_policy: PostPolicy,
        write_policy: MemoryWritePolicy,
        probe: FastMemoryProbe,
        expansion: MemoryExpansion,
        input_safety: InputSafety,
        output_safety: OutputSafety,
        llm: LLMBackend | None = None,
        coverage_fn: Callable[[State], Coverage] | None = None,
        coverage_threshold: float = 0.5,
        budget: Budget | None = None,
        logger: TrajectoryLogger | None = None,
    ) -> None:
        self.memory = memory
        self.retriever = retriever
        self.generator = generator
        self.pre = pre_policy
        self.post = post_policy
        self.write = write_policy
        self.probe = probe
        self.expansion = expansion
        self.in_safety = input_safety
        self.out_safety = output_safety
        # Bug #6: the LLM whose token usage is tracked into the Budget. Same
        # backend the policies and generator use; held as a reference so the
        # pipeline can sample `total_tokens_used` after each LLM-touching stage.
        self.llm = llm
        self.coverage_fn = coverage_fn or compute_coverage
        self.coverage_threshold = coverage_threshold
        self.default_budget = budget or Budget()
        self.logger = logger
        self._step_counter = 0

    # ---- public API ----

    def ask(self, question: str | Question, **kwargs) -> PipelineResult:
        if isinstance(question, str):
            question = Question(text=question, **kwargs)
        traj = Trajectory(question=question)
        state = State(question=question, budget=Budget(**self.default_budget.to_dict()))
        self._step_counter = 0

        # Reset LLM usage so per-episode budget counting starts from 0.
        # NB: this is a soft contract — if multiple Pipeline instances share
        # one LLMBackend, only one of them owns the counter at a time.
        if self.llm is not None:
            self.llm.reset_usage()
        baseline_tokens = self.llm.total_tokens_used if self.llm is not None else 0

        # 0. input safety
        verdict = self.in_safety(question.text)
        self._log(traj, state, "input_safety", None, {"allow": verdict.allow, "reason": verdict.reason})
        if not verdict.allow:
            return self._finalize(traj, state, blocked=True, reason=verdict.reason)

        # 1. probe
        state.probe = self.probe(question)
        self._log(traj, state, "probe", None, state.probe.to_dict())

        # 2-6. plan -> expand -> retrieve -> coverage -> verify (loop)
        self._run_loop(state, traj, baseline_tokens)

        # 7. generate
        if state.documents or state.memory_records:
            answer, tokens = self.generator.generate(
                state.effective_query, state.documents, state.memory_records
            )
            state.budget.tokens_spent += tokens
            state.answer = answer
            self._log(traj, state, "generate", None, {"answer_len": len(answer), "tokens": tokens})
        else:
            state.answer = (
                "I could not find sufficient evidence to answer that confidently."
            )
            self._log(traj, state, "generate", None, {"answer_len": len(state.answer), "tokens": 0})

        # Bug #6: sync remaining LLM tokens (policy + generator) into the budget
        # so downstream consumers (logs, RL) see the real usage.
        if self.llm is not None:
            state.budget.tokens_spent = max(
                state.budget.tokens_spent,
                self.llm.total_tokens_used - baseline_tokens,
            )

        state.finished = True

        # 8. memory write
        write_action = self.write.decide(state)
        self._log(traj, state, "write", write_action.to_dict(), None)
        if write_action.type == ActionType.WRITE:
            self._apply_write(state, write_action)

        # 9. output safety
        out_verdict = self.out_safety(state.answer or "")
        self._log(traj, state, "output_safety", None, {"allow": out_verdict.allow})
        if not out_verdict.allow:
            return self._finalize(traj, state, blocked=True, reason=out_verdict.reason)

        return self._finalize(traj, state)

    # ---- inner loop ----

    def _run_loop(self, state: State, traj: Trajectory, baseline_tokens: int = 0) -> None:
        while True:
            # pre-retrieval planning
            pre_action = self.pre.decide(state)
            self._sync_tokens(state, baseline_tokens)
            self._log(traj, state, "pre", pre_action.to_dict(), None)
            self._apply_pre(state, pre_action)

            # memory expansion (heavy read)
            records = self.expansion(state)
            state.memory_records = records
            self._log(traj, state, "expansion", None, {"n_records": len(records)})

            # retrieval — Bug #3: if Pre chose DECOMPOSE, run subqueries
            # individually and merge by max score. Single-query path otherwise.
            top_k = max(4, len(state.documents) if state.documents else 8)
            docs = self._retrieve_with_subqueries(state, top_k)

            # Bug #7 fix: if a provider filter produced zero docs, retry once
            # without the filter so a wrong CHOOSE_PROVIDER doesn't tank the
            # episode. We do NOT consume a retry attempt for this auto-recovery.
            if not docs and state.provider_filter:
                state.provider_filter = []
                docs = self._retrieve_with_subqueries(state, top_k)

            state.documents = docs
            self._log(
                traj,
                state,
                "retrieve",
                None,
                {
                    "n_docs": len(docs),
                    "doc_ids": [d.id for d in docs],
                    "sources": [d.source for d in docs],
                    # Bug #1 fix: embed truncated text so JSONL replay (and the
                    # LLMJudge) can actually verify factual claims.
                    "doc_texts": [
                        {
                            "id": d.id,
                            "source": d.source,
                            "score": d.score,
                            "text": (d.text or "")[:_OBS_DOC_TEXT_CAP],
                        }
                        for d in docs
                    ],
                },
            )

            # coverage (pluggable)
            state.coverage = self.coverage_fn(state)
            self._sync_tokens(state, baseline_tokens)
            state.coverage_history.append(state.coverage.score)
            self._log(traj, state, "coverage", None, state.coverage.to_dict())

            # post-retrieval verdict
            post_action = self.post.decide(state)
            self._sync_tokens(state, baseline_tokens)
            self._log(traj, state, "post", post_action.to_dict(), None)

            if post_action.type in (ActionType.ANSWER, ActionType.STOP):
                if post_action.type == ActionType.STOP:
                    state.answer = None
                break

            if post_action.type == ActionType.FILTER:
                drop = set(post_action.payload.get("drop") or [])
                state.documents = [d for d in state.documents if f"[{d.id}]" not in drop]
                state.coverage = self.coverage_fn(state)
                self._sync_tokens(state, baseline_tokens)
                # Bug #8 fix: do NOT append to coverage_history here — that
                # was corrupting coverage_delta() and the Budget check.
                if not state.budget.can_retry(state.attempt, state.coverage_delta()):
                    break
                continue

            # retry path: broaden / narrow / gap_retry → bump attempt
            state.attempt += 1
            self._apply_retry(state, post_action)
            if not state.budget.can_retry(state.attempt, state.coverage_delta()):
                break

    def _retrieve_with_subqueries(self, state: State, top_k: int) -> list[Document]:
        """Bug #3 fix: when subqueries are set, retrieve each then merge.

        Merge policy: dedupe by doc id; keep the highest score across runs.
        """
        queries: list[str] = [state.effective_query]
        if state.subqueries:
            queries.extend(state.subqueries)

        seen: dict[str, Document] = {}
        for q in queries:
            docs = self.retriever.retrieve(
                q,
                top_k=top_k,
                providers=state.provider_filter or None,
                expansion_terms=state.expansion_terms,
            )
            for d in docs:
                if d.id not in seen or d.score > seen[d.id].score:
                    seen[d.id] = d
        merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)
        return merged[:top_k]

    def _sync_tokens(self, state: State, baseline_tokens: int) -> None:
        """Bug #6: roll up cumulative LLM usage into the per-episode Budget."""
        if self.llm is None:
            return
        state.budget.tokens_spent = max(
            state.budget.tokens_spent,
            self.llm.total_tokens_used - baseline_tokens,
        )

    # ---- mutators ----

    def _apply_pre(self, state: State, action: Action) -> None:
        t = action.type
        p = action.payload
        if t == ActionType.NOOP:
            return
        if t == ActionType.REWRITE:
            state.rewritten_query = p.get("query") or state.rewritten_query
        elif t == ActionType.DECOMPOSE:
            subs = p.get("subqueries") or []
            state.subqueries = [s for s in subs if isinstance(s, str) and s.strip()]
        elif t == ActionType.INHERIT_FILTER:
            f = p.get("filters") or {}
            if isinstance(f, dict):
                state.inherited_filters.update(f)
                provider = f.get("provider")
                if provider and provider not in state.provider_filter:
                    state.provider_filter.append(provider)
        elif t == ActionType.CHOOSE_PROVIDER:
            providers = p.get("providers") or []
            for prov in providers:
                if prov and prov not in state.provider_filter:
                    state.provider_filter.append(prov)

    def _apply_retry(self, state: State, action: Action) -> None:
        t = action.type
        p = action.payload
        if t == ActionType.BROADEN:
            state.provider_filter = []
            state.expansion_terms = list(set(state.expansion_terms + ["overview", "general"]))
        elif t == ActionType.NARROW:
            state.expansion_terms = []
            if not state.provider_filter and state.documents:
                top = max(state.documents, key=lambda d: d.score)
                if top.source:
                    state.provider_filter = [top.source]
        elif t == ActionType.GAP_RETRY:
            aspect = p.get("aspect")
            if aspect:
                state.expansion_terms = list(set(state.expansion_terms + [aspect]))

    def _apply_write(self, state: State, action: Action) -> None:
        axes = action.payload.get("axes") or {}
        content = action.payload.get("content") or state.answer or ""
        keywords = action.payload.get("keywords") or []
        providers = action.payload.get("providers") or []
        evidence_ids = action.payload.get("evidence_ids") or []

        anchor_id: str | None = None
        if axes.get(WRITE_AXIS_FACTS):
            tags = ["fact"]
            if axes.get(WRITE_AXIS_ALIASES) and state.probe.coref_hints:
                tags.append("alias-resolved")
            rec = MemoryRecord(
                id=f"m-{uuid.uuid4().hex[:10]}",
                content=content,
                keywords=keywords,
                tags=tags,
                context=state.effective_query,
                metadata={
                    "session_id": state.question.session_id,
                    "user_id": state.question.user_id,
                    "providers": providers,
                    "evidence_ids": evidence_ids,
                    "coverage": state.coverage.score,
                },
            )
            anchor_id = self.memory.write(rec)

        if axes.get(WRITE_AXIS_PROVIDER_UTILITY):
            for prov in providers:
                self.memory.record_provider_utility(
                    state.question.session_id, prov, helped=True
                )

        if axes.get(WRITE_AXIS_REFLECTION) and anchor_id:
            self.memory.trigger_reflection(anchor_id)

    # ---- bookkeeping ----

    def _log(self, traj: Trajectory, state: State, stage: str, action, obs) -> None:
        self._step_counter += 1
        traj.append(
            TrajectoryStep(
                step=self._step_counter,
                stage=stage,
                state_before=state.snapshot(),
                action=action,
                observation=obs,
            )
        )

    def _finalize(
        self,
        traj: Trajectory,
        state: State,
        blocked: bool = False,
        reason: str = "",
    ) -> PipelineResult:
        traj.final_answer = state.answer
        traj.metadata["tokens_spent"] = state.budget.tokens_spent
        if blocked:
            traj.metadata["blocked"] = True
            traj.metadata["blocked_reason"] = reason
        if self.logger:
            self.logger.write(traj)
        return PipelineResult(
            answer=state.answer,
            trajectory=traj,
            state=state,
            blocked=blocked,
            blocked_reason=reason,
        )


# ----- convenience builder -----

def build_default_pipeline(
    config: Config | None = None,
    logger: TrajectoryLogger | None = None,
) -> Pipeline:
    """Construct a pipeline from Config, defaulting to safe in-memory backends."""
    cfg = config or Config.from_env()

    # memory
    if cfg.memory.backend == "amem":
        from security_agent.memory.amem_adapter import AMemAdapter

        memory: MemoryStore = AMemAdapter(
            cfg.memory.amem_root,
            llm_backend=("openai" if cfg.llm.backend != "dummy" else "openai"),
            llm_model=cfg.llm.model,
            api_key=cfg.llm.api_key or "sk-noop",
        )
    else:
        memory = InMemoryStore()

    # retriever
    if cfg.retriever.backend == "handmade_rag":
        from security_agent.retrieval.handmade_rag_adapter import HandmadeRagAdapter

        retriever: Retriever = HandmadeRagAdapter(cfg.retriever.handmade_rag_root)
    else:
        retriever = DummyRetriever()

    # llm + generator
    llm: LLMBackend = build_llm(cfg.llm)
    generator = LLMGenerator(llm)

    # policies
    pre = pre_pkg.build(cfg.pipeline.pre_policy, llm=llm)
    post = post_pkg.build(
        cfg.pipeline.post_policy,
        llm=llm,
        threshold=cfg.pipeline.coverage_threshold,
    )
    write = write_pkg.build(cfg.pipeline.write_policy, llm=llm)

    probe = FastMemoryProbe(memory, recent_n=cfg.memory.probe_recent_turns)
    expansion = MemoryExpansion(
        memory,
        top_k=cfg.memory.expansion_top_k,
        kg_hops=cfg.memory.kg_hops,
    )

    in_safety = InputSafety(cfg.safety.block_patterns) if cfg.safety.enable_input else InputSafety(())
    out_safety = OutputSafety(cfg.safety.block_patterns) if cfg.safety.enable_output else OutputSafety(())

    budget = Budget(
        max_retries=cfg.budget.max_retries,
        max_tokens=cfg.budget.max_tokens,
        min_coverage_delta=cfg.budget.min_coverage_delta,
    )

    coverage_fn = build_coverage(cfg.pipeline.coverage_kind, llm=llm)

    return Pipeline(
        memory=memory,
        retriever=retriever,
        generator=generator,
        pre_policy=pre,
        post_policy=post,
        write_policy=write,
        probe=probe,
        expansion=expansion,
        input_safety=in_safety,
        output_safety=out_safety,
        llm=llm,
        coverage_fn=coverage_fn,
        coverage_threshold=cfg.pipeline.coverage_threshold,
        budget=budget,
        logger=logger,
    )
