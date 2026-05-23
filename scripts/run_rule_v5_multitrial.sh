#!/usr/bin/env bash
# Run 2 more Rule-on-v5 trials sequentially. Trial 1 already exists at
# experiments/baseline_2026_05_22_v5/bench50_rule_v5.jsonl.
#
# Different session_prefix per trial so trajectory IDs are distinct and
# stochastic state (any internal randomness in DeepSeek sampling, BGE
# rerank ties) doesn't reproduce identical answers.

set -eu

ROOT=/mnt/d/Project/WORK/security-agent
BENCH=/mnt/d/Project/WORK/all-in-rag/data/benchmarks/security_benchmark_saq.jsonl
OUT_DIR=$ROOT/experiments/baseline_2026_05_22_v5
PY=/home/wyt/miniconda3/envs/all-in-rag/bin/python

export SA_RETRIEVER_BACKEND=handmade_rag
export SA_HANDMADE_RAG_INDEX_DIR=/mnt/d/Project/WORK/all-in-rag/code/handmade_rag/artifacts/security_index_v5
export OPENAI_API_KEY=${OPENAI_API_KEY:?set DeepSeek key in env}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.deepseek.com}
export SA_LLM_MODEL=${SA_LLM_MODEL:-deepseek-chat}
export SA_LLM_BASE_URL=$OPENAI_BASE_URL
export SA_LLM_API_KEY=$OPENAI_API_KEY
export SA_LLM_BACKEND=openai
export RAG_REDIS_RETRIEVAL_CACHE_ENABLED=false
export RAG_REDIS_RERANK_CACHE_ENABLED=false
export RAG_REDIS_ANSWER_CACHE_ENABLED=false
export RAG_GENERATION_BACKEND=${RAG_GENERATION_BACKEND:-openai}
export RAG_GENERATION_API_KEY=$OPENAI_API_KEY
export RAG_GENERATION_BASE_URL=$OPENAI_BASE_URL
export RAG_GENERATION_MODEL=${RAG_GENERATION_MODEL:-deepseek-chat}

cd "$ROOT"
for trial in 2 3; do
    OUT_FILE=$OUT_DIR/bench50_rule_v5_t${trial}.jsonl
    echo "=== Rule v5 trial ${trial} ==="
    $PY scripts/run_benchmark.py "$BENCH" \
        --policy rule \
        --out "$OUT_FILE" \
        --session-prefix "rule-v5-t${trial}" \
        --judge llm \
      2>&1 | tee "$OUT_DIR/bench50_rule_v5_t${trial}.log"
    echo
    echo "=== Rule v5 trial ${trial} done ==="
done
