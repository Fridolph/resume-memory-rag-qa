# resume-memory-rag-qa

一个适合学习的简历 Memory / RAG / QA 渐进式 demo。

这个仓库从最基础的：

- 简历结构化解析
- Milvus 向量入库
- 基础问答链路

逐步演进到：

- 检索优化
- Pipeline 重构
- 会话存储
- Prompt 目录化
- 配置外置
- 主证据优先
- 流式进度反馈

## 当前目录

- `src/`：源码
- `drafts/`：对应每一阶段的中文学习文档

## 快速开始

```bash
pnpm install
cp .env.example .env
pnpm run ingest
pnpm run ask -- "这个候选人有哪些 AI Agent 开发相关经验？"
```
