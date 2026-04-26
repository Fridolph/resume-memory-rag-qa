# resume-memory-rag-qa

一个用于教学和文章配套的 **简历 Memory / RAG / QA 渐进式 Demo**。

这个仓库不是一个追求“开箱即用产品化”的完整应用，而是一个保留学习痕迹的实验仓：它按 `v1 → v7` 保存了简历 RAG 从基础问答到证据分层、去噪、流式输出的演进过程，适合配合文章阅读和本地运行验证。

## 项目定位

这个项目主要回答三个问题：

1. 如何把一份 Markdown 简历拆成适合 RAG 的结构化 chunks？
2. 如何从“能召回内容”逐步优化到“能选对证据”？
3. 如何把实验脚本整理成可迁移到 `my-resume` 项目的 RAG 能力雏形？

如果你正在学习：

- RAG 基础链路
- Embedding + Milvus 向量检索
- Prompt 与证据组织
- 简历/个人知识库问答
- JS/Node.js 调用 OpenAI-compatible API

这个仓库可以作为一个比较小、但版本演进完整的阅读样例。

## 技术栈

- Runtime：Node.js ESM
- Package Manager：pnpm
- LLM / Embedding：OpenAI-compatible API
- Vector DB：Milvus / Zilliz Milvus SDK
- RAG 编排：原生 JS 脚本渐进式实现
- 配置：`.env` + JSON config

## 目录结构

```text
.
├── src/
│   ├── inspect-resume.mjs                 # 查看 Markdown 简历解析结果
│   ├── ingest-resume.mjs                  # 生成 chunks / embeddings 并写入 Milvus
│   ├── ask-resume*.mjs                    # v1-v7 问答入口
│   ├── embedding-client.mjs               # embedding 请求封装
│   ├── chat-model-client.mjs              # DeepSeek v4 Chat 专用入口
│   ├── resume-parser.mjs                  # Markdown 简历解析与 chunk 构建
│   ├── rag/                               # v3 pipeline 初版模块
│   ├── rag4/                              # v4 会话、prompt loader 等
│   ├── rag5/                              # v5 配置外置与 context-builder
│   ├── rag6/                              # v6 主证据优先与精细去噪
│   └── rag7/                              # v7 完整闭合版本
├── docker-compose.milvus.yml              # 本地 Milvus standalone 开发环境
├── drafts/                                # 每一阶段的中文学习记录
├── articles/                              # 系列文章草稿/整理稿
├── package.json
└── .env.example
```

## 前置依赖

运行本项目需要：

- Node.js LTS
- pnpm
- Docker / Docker Compose
- 可用的 Chat 模型 API Key
- 可用的 Embedding 模型 API Key

### 1. 安装 Node.js 与 pnpm

建议使用较新的 Node.js LTS 版本。当前项目使用 ESM，推荐先确认：

```bash
node -v
pnpm -v
```

如果本机没有 pnpm，可以安装：

```bash
npm install -g pnpm
```

### 2. 安装 Docker

本地运行 Milvus 需要 Docker。推荐安装 Docker Desktop：

- macOS / Windows：安装 Docker Desktop
- Linux：安装 Docker Engine 与 Docker Compose Plugin

确认命令可用：

```bash
docker --version
docker compose version
```

### 3. 安装项目依赖

```bash
pnpm install
```

### 4. 启动本地 Milvus

本项目默认连接：

```text
localhost:19530
```

仓库内置了一个 Milvus standalone 的 Docker Compose 配置：

```bash
pnpm run milvus:up
```

查看容器状态：

```bash
docker compose -f docker-compose.milvus.yml ps
```

查看 Milvus 日志：

```bash
pnpm run milvus:logs
```

停止本地 Milvus：

```bash
pnpm run milvus:down
```

本地数据会写入 `.runtime/milvus/`，该目录已被 git 忽略。

如果你不想使用本地 Docker，也可以改用：

- Zilliz Cloud
- 其他兼容 Milvus SDK 的服务

如果使用远程地址，在 `.env` 中配置：

```env
MILVUS_ADDRESS=your-milvus-host:19530
```

## 环境变量

复制示例配置：

```bash
cp .env.example .env
```

然后按你的模型服务填写 `.env`。

### 基础 Chat 模型

普通 `ask:v1` 到 `ask:v7` 默认读取：

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
MODEL_NAME=
```

### DeepSeek v4 专用入口

如果运行：

```bash
pnpm run ask:v7:dpv4
```

需要配置：

```env
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_NAME=deepseek-v4-flash
```

注意：DeepSeek v4 这里只用于 Chat / Answer 生成。

### Embedding 模型

Embedding 必须和写入 Milvus 时使用的模型保持一致。

```env
EMBEDDINGS_API_KEY=
EMBEDDINGS_URL=
EMBEDDINGS_MODEL_NAME=
```

例如你使用 GLM embedding 写入了 1024 维向量，那么查询时也必须继续使用同一个 embedding 模型；否则需要重新 `ingest`。

> DeepSeek 官方 API 当前没有 embeddings 端点，不能把 `EMBEDDINGS_URL` 配成 `https://api.deepseek.com`。

## 快速开始

下面是一条从零开始的本地运行链路：

```bash
pnpm install
cp .env.example .env
pnpm run milvus:up
pnpm run inspect
pnpm run ingest
pnpm run ask:v7 -- "这个候选人有哪些 AI Agent 开发相关经验？"
```

其中 `.env` 需要你先填好 Chat / Embedding 相关配置。

### 1. 查看解析结果

建议先不要急着写入向量库，先查看简历会被拆成什么结构：

```bash
pnpm run inspect
```

这一步用于确认：

- section 是否识别正确
- 工作经历 / 项目经历是否拆分合理
- chunk 内容是否适合进入 RAG

### 2. 写入 Milvus

```bash
pnpm run ingest
```

这个命令会完成：

```text
解析 Markdown 简历
  ↓
生成结构化 records
  ↓
生成 chunks
  ↓
调用 embedding 模型
  ↓
写入 Milvus collection
```

### 3. 运行最新 v7 问答

```bash
pnpm run ask:v7 -- "这个候选人有哪些 AI Agent 开发相关经验？"
```

也可以使用默认入口：

```bash
pnpm run ask -- "这个候选人有哪些 AI Agent 开发相关经验？"
```

默认 `pnpm run ask` 指向当前最新的 `v7`。

### 4. 使用 DeepSeek v4 Chat 版本

```bash
pnpm run ask:v7:dpv4 -- "这个候选人有哪些 AI Agent 开发相关经验？"
```

这个入口只替换回答生成模型，RAG 主流程和 embedding 索引保持不变。

## 版本脚本

```bash
pnpm run ask:v1 -- "问题"
pnpm run ask:v2 -- "问题"
pnpm run ask:v3 -- "问题"
pnpm run ask:v4 -- "问题"
pnpm run ask:v5 -- "问题"
pnpm run ask:v6 -- "问题"
pnpm run ask:v7 -- "问题"
pnpm run ask:v7:dpv4 -- "问题"
```

每个版本都保留了一次具体学习目标：

| 版本 | 重点 |
|---|---|
| v1 | 基础向量召回与问答链路 |
| v2 | 扩大候选召回、简单 rerank、回答策略约束 |
| v3 | 抽出 retriever / prompt-builder / rag-pipeline |
| v4 | 会话存储、Prompt 目录化、噪声过滤 |
| v5 | 配置外置、context-builder 拆分 |
| v6 | 主证据优先、精细去噪 |
| v7 | 跨区补证、reserve 兜底、流式输出、版本内闭合 |
| v7:dpv4 | v7 + DeepSeek v4 Chat 生成入口 |

## 推荐阅读顺序

建议按 `drafts/` 顺序阅读，因为每一篇都对应一次真实问题和代码演进：

1. `drafts/01-简历RAG字段拆分与存储设计.md`
2. `drafts/02-简历RAG检索针对性与回答质量优化记录.md`
3. `drafts/03-简历RAG管线化重构与可复用设计.md`
4. `drafts/04-简历RAG版本4会话存储Prompt目录化与噪声过滤设计.md`
5. `drafts/05-简历RAG版本5策略修正配置外置与上下文构建拆分.md`
6. `drafts/06-简历RAG版本6主证据优先与精细去噪优化.md`
7. `drafts/07-简历RAG版本7跨区补证与流式进度反馈优化.md`

如果只想看最终完整版本，可以从这些文件开始：

- `src/ask-resume-7-progressive-ask.mjs`
- `src/rag7/rag-pipeline-v7.mjs`
- `src/rag7/context-builder-v7.mjs`
- `src/rag7/prompt-builder-v7.mjs`
- `src/rag7/config/`

## 常见问题

### 1. 为什么换 Chat 模型后 embedding 报错？

RAG 中有两类模型：

```text
Chat 模型：负责最终回答
Embedding 模型：负责生成向量并检索 Milvus
```

Chat 模型可以相对轻量替换，但 embedding 模型决定了向量坐标系。

如果你换了 embedding 模型，需要重新运行：

```bash
pnpm run ingest
```

### 2. 为什么查询向量维度和 collection 不一致？

这通常说明：

- 写入 Milvus 时用的是一个 embedding 模型；
- 查询时换成了另一个 embedding 模型；
- 或者环境变量指向了不同 provider。

请确认 `EMBEDDINGS_MODEL_NAME` 和写入时一致。

### 3. 为什么 DeepSeek v4 不用于 embedding？

DeepSeek v4 在这个项目中只用于 Chat 生成。DeepSeek 官方 API 当前没有 embeddings 端点，因此不能直接作为本项目的 embedding provider。

### 4. 为什么有这么多版本脚本？

这是一个教学仓库，不是只保留最终实现的业务项目。

版本脚本的目的，是让读者可以按演进顺序看到：

```text
问题出现
  ↓
临时解决
  ↓
结构化重构
  ↓
形成更稳定版本
```

## 配套文章

这个仓库对应「JS全栈AI学习」系列中的简历 RAG 实战三篇：

1. 「JS全栈AI学习」十二、从 Prompt 到 RAG：把 Markdown 简历变成可检索知识库
2. 「JS全栈AI学习」十三、从能回答到答得准：给简历 RAG 加上重排、去噪和主证据优先
3. 「JS全栈AI学习」十四、从实验脚本到产品雏形：简历 RAG v7 的一次完整收口

## License

MIT
