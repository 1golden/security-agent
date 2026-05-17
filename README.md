# security-agent

一个可替换策略层的 Agent + RAG + Memory 框架，面向安全/合规问答场景，但架构通用。

```
Question
  ↓
Pre-Retrieval Policy       (rewrite / decompose / no-op)
  ↓
Memory Read                (含 KG 多跳)
  ↓
RAG Retrieval + Rerank
  ↓
Coverage Check
  ↓
Post-Retrieval Policy      (filter / retry / stop / answer)
  ↓ (若 retry → 回到 Pre-Retrieval Policy)
Answer Generation
  ↓
Memory Write Policy        (写 / 不写 / 触发演化)
  ↓
Memory Write + Eval Log
```

## 三种 Policy 实现，可替换

| 实现 | 用途 | 状态 |
|------|------|------|
| `RulePolicy`  | 启发式规则 baseline | ✅ 可用 |
| `LLMPolicy`   | 轻量 ReAct / planner，LLM 即时决策 | ✅ 可用 |
| `RLPolicy`    | 用收集的 trajectory 训练 (GRPO/PPO) | 🚧 接口对齐，待训练 |

三类 policy（pre/post/write）独立可替换，例如：
- 全 Rule = 经典 RAG baseline
- pre/post = LLM，write = Rule（兼顾质量与可控成本）
- 三者全 RL = 未来目标

## 快速开始

```bash
cd security-agent
pip install -e .
cp .env.example .env   # 填 LLM key (或不填，使用 dummy 后端)

# 端到端 demo（dummy backend，无需 API key）
python scripts/run_demo.py

# 收集 trajectory，供 RL 训练
python scripts/collect_trajectories.py --n 100 --out runs/traj_v1.jsonl
```

## 项目结构

```
src/security_agent/
├── pipeline.py            # 主编排器（while-loop with retry）
├── types.py               # Question / State / Action / Trajectory
├── config.py              # 全局配置 dataclass
├── coverage.py            # 覆盖度检查
├── policies/
│   ├── base.py            # Policy 抽象接口
│   ├── pre_retrieval/     # Rule / LLM / RL
│   ├── post_retrieval/    # Rule / LLM / RL
│   └── memory_write/      # Rule / LLM / RL
├── memory/
│   ├── base.py            # MemoryStore 接口
│   ├── inmem.py           # 进程内简易实现（测试 / demo）
│   └── amem_adapter.py    # 包装 all-in-rag/A-mem-sys
├── retrieval/
│   ├── base.py            # Retriever 接口
│   ├── dummy.py           # mock retriever
│   └── handmade_rag_adapter.py  # 包装 all-in-rag/code/handmade_rag
├── generation/
│   └── llm_generator.py
└── eval_log/
    ├── logger.py          # JSONL trajectory writer
    └── replay.py          # 回放 / RL 数据准备
```

## 设计要点

1. **Memory Read 在 Policy 之后**：Pre-Retrieval Policy 先 rewrite/decompose 才查记忆，避免拿原始模糊 query 检索（参考 RMM, ACL 2025）。
2. **Memory Write 是主动决策**：不是无脑追加，由 `MemoryWritePolicy` 决定写入粒度、是否触发演化（参考 A-Mem 笔记演化机制）。
3. **Pre / Post 策略对称**：检索前做规划，检索后做验证，奖励信号分离便于 RL 训练。
4. **每一步都进 Trajectory**：完整中间状态写入 JSONL，足以做 behavior cloning 和 step-wise reward attribution。

详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 与相邻项目的关系

- **数据/记忆源**：可包装 `all-in-rag/A-mem-sys`（A-Mem 实现）作为长期记忆后端。
- **检索后端**：可包装 `all-in-rag/code/handmade_rag`（六阶段安全 RAG）作为 Retriever。
- 这两项都不是硬依赖，框架自带 in-memory + dummy 实现可独立跑通。

## License

MIT
