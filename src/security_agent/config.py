"""Centralized configuration. Reads from env vars with sensible defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass
class LLMConfig:
    backend: str = field(default_factory=lambda: _env("SA_LLM_BACKEND", "dummy"))
    model: str = field(default_factory=lambda: _env("SA_LLM_MODEL", "gpt-4o-mini"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    base_url: str | None = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL"))
    temperature: float = field(default_factory=lambda: _env_float("SA_LLM_TEMP", 0.2))
    max_tokens: int = field(default_factory=lambda: _env_int("SA_LLM_MAX_TOKENS", 1024))


@dataclass
class MemoryConfig:
    backend: str = field(default_factory=lambda: _env("SA_MEMORY_BACKEND", "inmem"))
    amem_root: str = field(
        default_factory=lambda: _env(
            "SA_AMEM_ROOT", "/mnt/d/Project/WORK/all-in-rag/A-mem-sys"
        )
    )
    chroma_path: str = field(default_factory=lambda: _env("SA_CHROMA_PATH", "./chroma_db"))
    probe_recent_turns: int = field(default_factory=lambda: _env_int("SA_PROBE_TURNS", 4))
    expansion_top_k: int = field(default_factory=lambda: _env_int("SA_MEMORY_TOPK", 5))
    kg_hops: int = field(default_factory=lambda: _env_int("SA_MEMORY_KG_HOPS", 2))


@dataclass
class RetrieverConfig:
    backend: str = field(default_factory=lambda: _env("SA_RETRIEVER_BACKEND", "dummy"))
    handmade_rag_root: str = field(
        default_factory=lambda: _env(
            "SA_HANDMADE_RAG_ROOT", "/mnt/d/Project/WORK/all-in-rag/code/handmade_rag"
        )
    )
    top_k: int = field(default_factory=lambda: _env_int("SA_RETRIEVER_TOPK", 8))
    rerank_top_k: int = field(default_factory=lambda: _env_int("SA_RETRIEVER_RERANK_TOPK", 4))


@dataclass
class SafetyConfig:
    """Input / output safety wrappers around the pipeline."""

    enable_input: bool = field(default_factory=lambda: _env_bool("SA_SAFETY_IN", True))
    enable_output: bool = field(default_factory=lambda: _env_bool("SA_SAFETY_OUT", True))
    block_patterns: tuple[str, ...] = (
        # crude defaults; real deployment should plug in Llama Guard / shieldgemma
        "ignore previous instructions",
        "system:\nyou are",
    )


@dataclass
class BudgetConfig:
    max_retries: int = field(default_factory=lambda: _env_int("SA_MAX_RETRIES", 2))
    max_tokens: int = field(default_factory=lambda: _env_int("SA_MAX_TOKENS", 8000))
    min_coverage_delta: float = field(
        default_factory=lambda: _env_float("SA_MIN_COV_DELTA", 0.05)
    )


@dataclass
class PipelineConfig:
    coverage_threshold: float = field(
        default_factory=lambda: _env_float("SA_COVERAGE_THRESHOLD", 0.5)
    )
    log_dir: str = field(default_factory=lambda: _env("SA_LOG_DIR", "runs"))
    pre_policy: str = field(default_factory=lambda: _env("SA_PRE_POLICY", "rule"))   # rule|llm|rl
    post_policy: str = field(default_factory=lambda: _env("SA_POST_POLICY", "rule")) # rule|llm|rl
    write_policy: str = field(default_factory=lambda: _env("SA_WRITE_POLICY", "rule"))


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_env(cls) -> "Config":
        return cls()
