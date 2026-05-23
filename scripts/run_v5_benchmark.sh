#!/usr/bin/env bash
# v5 benchmark — run all three policies on the 50-Q SAQ benchmark using
# the security_index_v5 (extended with NIST SP 800-53 + tactic sub-route).
#
# Outputs go to experiments/baseline_2026_05_22_v5/
set -eu

ROOT=/mnt/d/Project/WORK/security-agent
BENCH=/mnt/d/Project/WORK/all-in-rag/data/benchmarks/security_benchmark_saq.jsonl
OUT_DIR=$ROOT/experiments/baseline_2026_05_22_v5
PY=/home/wyt/miniconda3/envs/all-in-rag/bin/python

export SA_RETRIEVER_BACKEND=handmade_rag
export SA_HANDMADE_RAG_INDEX_DIR=/mnt/d/Project/WORK/all-in-rag/code/handmade_rag/artifacts/security_index_v5

# Use the same DeepSeek backend as v4.
export OPENAI_API_KEY=${OPENAI_API_KEY:?set DeepSeek key in env}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.deepseek.com}
export SA_LLM_MODEL=${SA_LLM_MODEL:-deepseek-chat}
export SA_LLM_BASE_URL=$OPENAI_BASE_URL
export SA_LLM_API_KEY=$OPENAI_API_KEY
export SA_LLM_BACKEND=openai

# Disable handmade_rag's Redis caches so we don't read stale v4 chunks.
export RAG_REDIS_RETRIEVAL_CACHE_ENABLED=false
export RAG_REDIS_RERANK_CACHE_ENABLED=false
export RAG_REDIS_ANSWER_CACHE_ENABLED=false

# Generation backend in handmade_rag: prefer OpenAI-compatible (DeepSeek).
export RAG_GENERATION_BACKEND=${RAG_GENERATION_BACKEND:-openai}
export RAG_GENERATION_API_KEY=$OPENAI_API_KEY
export RAG_GENERATION_BASE_URL=$OPENAI_BASE_URL
export RAG_GENERATION_MODEL=${RAG_GENERATION_MODEL:-deepseek-chat}

mkdir -p "$OUT_DIR"

POLICY=${1:?usage: $0 <rule|llm|router>}
OUT_FILE=$OUT_DIR/bench50_${POLICY}_v5.jsonl

cd "$ROOT"
$PY scripts/run_benchmark.py "$BENCH" \
    --policy "$POLICY" \
    --out "$OUT_FILE" \
    --session-prefix "${POLICY}-v5" \
    --judge llm \
  2>&1 | tee "$OUT_DIR/bench50_${POLICY}_v5.log"

echo
echo "=== $POLICY v5 finished. Results: $OUT_FILE ==="
