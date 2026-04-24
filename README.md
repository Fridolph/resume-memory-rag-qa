# resume-memory-rag-qa

一个适合学习的简历 Memory / RAG / QA 渐进式 demo 仓库。

这个仓库按 **v1 → v7** 的顺序保留了演进过程，方便学习者按提交记录理解：

1. 基础解析、入库与初版问答
2. 检索针对性优化
3. RAG Pipeline 重构
4. 会话存储、Prompt 目录化、噪声过滤
5. 配置外置、context-builder 拆分
6. 主证据优先与精细去噪
7. 跨区补证、阶段日志与流式回答

## 目录说明

- `src/`：源码，保留每一代 ask 脚本与对应子模块
- `drafts/`：每一阶段对应的中文学习文档

## 推荐阅读顺序

- `drafts/01-简历RAG字段拆分与存储设计.md`
- `drafts/02-简历RAG检索针对性与回答质量优化记录.md`
- `drafts/03-简历RAG管线化重构与可复用设计.md`
- `drafts/04-简历RAG版本4会话存储Prompt目录化与噪声过滤设计.md`
- `drafts/05-简历RAG版本5策略修正配置外置与上下文构建拆分.md`
- `drafts/06-简历RAG版本6主证据优先与精细去噪优化.md`
- `drafts/07-简历RAG版本7跨区补证与流式进度反馈优化.md`

## 快速开始

```bash
pnpm install
cp .env.example .env
pnpm run ingest
pnpm run ask -- "这个候选人有哪些 AI Agent 开发相关经验？"
```

## 版本脚本

```bash
pnpm run ask:v1 -- "问题"
pnpm run ask:v2 -- "问题"
pnpm run ask:v3 -- "问题"
pnpm run ask:v4 -- "问题"
pnpm run ask:v5 -- "问题"
pnpm run ask:v6 -- "问题"
pnpm run ask:v7 -- "问题"
```

默认 `pnpm run ask` 指向当前最新的 `v7`。
