<div align="center">

# 🏥 OpenOE-Lite

**一个 API Key、零向量库、零本地语料<br>直接从 2.5 亿篇开放学术论文中生成循证医学回答**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAlex](https://img.shields.io/badge/data-OpenAlex-orange)](https://openalex.org)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-blueviolet)](https://openrouter.ai)

完全开源地轻量级复现 [OpenEvidence](https://www.openevidence.com/) 风格的医疗循证问答体验。<br>
不建索引，不存论文，不训模型，一个API就跑起来 — 用查询时编排替代预处理重资产。

💡 **为什么是 Lite？** — 除了一个 LLM API，仅依赖 [OpenAlex](https://openalex.org) 的 2.5 亿篇开放学术数据。无需向量数据库、无需 Embedding 模型、无需下载处理论文，略过昂贵繁重的数据集处理和embedding，即刻跑通。

[中文](#中文) · [English](#english)

<img src="imgs/screenshot.png" alt="OpenOE-Lite Screenshot" width="900">

</div>

---

<a id="中文"></a>

## 🇨🇳 中文

### ✨ 特点

- 🪶 **无需本地向量数据库** — 无本地语料、无 Embedding 预处理、无索引构建。基于 [OpenAlex](https://openalex.org) 2.5 亿篇开放学术文献实时检索，一个 API Key 开箱即用
- 🔍 **跨语言三视角检索** — 中文问题自动生成：英文学术检索式 + 自然语言改写 + HyDE 假想摘要，三条路径同时召回，用查询增强替代本地索引
- 🎯 **小模型证据门控** — 廉价推理模型并发审读每篇候选，将无关论文挡在大模型上下文之外。看似多一步，实则总 token 更少、延迟更低、回答更准
- 📝 **Grounded Answer** — 循证结构化回答，引用自动重编号并与源文献对齐，每个结论可溯源到具体论文
- 🧩 **Lite 内核，Full 上限** — 统一文档模型 + 六阶段编排 + 可插拔数据源，从零索引平滑升级到本地向量库 + 临床指南的完整 RAG

> *「一番の近道は遠回りだった。遠回りこそが俺の最短の道だった」*
> — Gyro Zeppeli, **Lesson 5** (《JOJO的奇妙冒险 第七部 飙马野郎》)
>
> 小模型全文过滤看似绕了远路，但把噪音挡在大模型上下文之外，节省了回答模型的昂贵context window，反而省时又省钱。

### 🏗️ 管线架构

```
用户提问（中文 / 英文）
  │
  ├─ Stage 0: 安全边界          轻量 LLM 拦截非医学/危险问题
  ├─ Stage 1: 三视角查询增强     检索式生成 + 英文改写 + HyDE 假想摘要
  ├─ Stage 2: 开放学术检索       三策略从 2.5 亿篇论文中召回候选
  ├─ Stage 3: 去重与排序         DOI/标题模糊去重 + RRF 多路融合
  ├─ Stage 4: 小模型证据门控     廉价模型并发审读，砍掉噪音 ← 真正的捷径
  └─ Stage 5: 循证回答生成       强模型精读 + 引用对齐 + 参考文献格式化
```

### 🚀 快速开始

```bash
git clone https://github.com/yourname/openoe-lite.git
cd openoe-lite
pip install -e .
cp .env_template .env   # 编辑：填入 OPENROUTER_API_KEY
python -m openoe        # 浏览器打开 http://localhost:8000
```

### ⚙️ 配置

编辑 `openoe/config/pipeline.yaml` 自定义模型和各阶段参数。  
所有模型通过 [OpenRouter](https://openrouter.ai/models) 调用，直接替换模型名即可切换。

| 环境变量 | 必填 | 说明 |
|---------|:----:|------|
| `OPENROUTER_API_KEY` | ✅ | [获取 Key](https://openrouter.ai/keys) |
| `OPENALEX_EMAIL` | | OpenAlex 礼貌池邮箱 |
| `OPENOE_DEBUG` | | 设为 `true` 记录 LLM 完整请求/响应到 `logs/llm/` |

### 📡 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 🖥️ Web 前端 |
| `POST` | `/api/query` | RAG 问答 |
| `WS` | `/ws/query` | 实时问答 (带进度推送) |
| `GET` | `/api/sources` | 检索路线列表 |
| `GET` | `/api/config` | 当前配置 (脱敏) |
| `GET` | `/api/traces/{id}` | 执行轨迹 |
| `GET` | `/health` | 健康检查 |

### 🧩 Lite → Full 升级路线

模块化架构让你可以逐步增强：

- 🗄️ 接入本地论文数据库 (PostgreSQL + pgvector 全文/向量检索)
- 📚 添加临床指南数据库 (chunk 级向量检索 + 段落提取)
- 📥 构建 PubMed XML 数据灌入管线
- 🤖 实现 Agentic RAG (多轮迭代检索)
- 👤 添加用户系统 + 查询历史

---

<a id="english"></a>

## 🌐 English

### ✨ Features

- 🪶 **No Local Vector DB Needed** — No local corpus, no embedding preprocessing, no index building. Real-time retrieval from [OpenAlex](https://openalex.org)'s 250M+ open academic works — one API key, ready to go
- 🔍 **Cross-Lingual 3-View Retrieval** — Chinese questions auto-generate: English search expression + natural language rewrite + HyDE hypothetical abstract. Three paths retrieve simultaneously, replacing local indexing with query-time augmentation
- 🎯 **Small-Model Evidence Gating** — Cheap reasoning models concurrently review every candidate, keeping irrelevant papers out of the strong model's context. One extra step → fewer tokens, lower latency, better answers
- 📝 **Grounded Answers** — Evidence-based structured responses with auto-reindexed citations aligned to source papers. Every conclusion is traceable
- 🧩 **Lite Core, Full Ceiling** — Unified document model + 6-stage orchestrator + pluggable sources. Upgrade smoothly from zero-index to full RAG with local vector DB + clinical guidelines

> *"The shortest route was a detour. It was a detour that was our shortest path."*
> — Gyro Zeppeli, **Lesson 5** (*JoJo's Bizarre Adventure Part 7: Steel Ball Run*)
>
> Small-model filtering looks like a detour — but keeping noise out of the strong model's expensive context window actually saves both time and money.

### 🏗️ Architecture

```
Question (Chinese / English)
  │
  ├─ Stage 0: Safety Gate           lightweight LLM blocks non-medical / unsafe queries
  ├─ Stage 1: 3-View Enhancement    search expression + English rewrite + HyDE hypothetical abstract
  ├─ Stage 2: Open Academic Search   3 strategies retrieve from 250M+ papers
  ├─ Stage 3: Dedup & Rank          DOI/title fuzzy dedup + RRF multi-path fusion
  ├─ Stage 4: Evidence Gating       cheap models concurrent review ← the real shortcut
  └─ Stage 5: Grounded Answer       strong model deep-read + citation alignment + reference formatting
```

### 🚀 Quick Start

```bash
git clone https://github.com/yourname/openoe-lite.git
cd openoe-lite
pip install -e .
cp .env_template .env   # Edit: add OPENROUTER_API_KEY
python -m openoe        # Open http://localhost:8000
```

### ⚙️ Configuration

Edit `openoe/config/pipeline.yaml` to customize models and stages.  
Models are accessed via [OpenRouter](https://openrouter.ai/models) — swap any model by name.

| Variable | Required | Description |
|----------|:--------:|-------------|
| `OPENROUTER_API_KEY` | ✅ | [Get your key](https://openrouter.ai/keys) |
| `OPENALEX_EMAIL` | | Email for OpenAlex polite pool |
| `OPENOE_DEBUG` | | `true` to log LLM requests to `logs/llm/` |

### 📡 API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | 🖥️ Web frontend |
| `POST` | `/api/query` | RAG query |
| `WS` | `/ws/query` | Real-time query with progress |
| `GET` | `/api/sources` | Active retrieval routes |
| `GET` | `/api/config` | Current config (redacted) |
| `GET` | `/api/traces/{id}` | Execution trace |
| `GET` | `/health` | Health check |

### 📁 Project Structure

```
openoe/
├── pipeline/          # 6 RAG stages (one file each)
│   ├── safety.py      #   Stage 0: Safety classification
│   ├── enhance.py     #   Stage 1: HyDE + translate + keywords
│   ├── retrieval.py   #   Stage 2: Multi-source concurrent retrieval
│   ├── merge.py       #   Stage 3: RRF fusion + dedup
│   ├── filter.py      #   Stage 4: LLM relevance filter
│   ├── answer.py      #   Stage 5: Answer + citation formatting
│   └── orchestrator.py#   Wires stages 0→5 with progress callbacks
├── sources/
│   └── openalex.py    # OpenAlex 3-way search
├── llm/
│   ├── client.py      # Async LLM client (retry, reasoning control)
│   └── prompts.py     # YAML prompt loading
├── core/              # Config, models, tracing
├── api/app.py         # FastAPI + WebSocket + frontend serving
├── static/index.html  # Frontend (single-file, no build step)
├── config/            # pipeline.yaml + prompts.yaml
└── utils/             # Text cleaning + dedup
```

### 🧩 Lite → Full Upgrade Path

The modular architecture lets you progressively enhance:

- 🗄️ Add a local paper database (PostgreSQL + pgvector for full-text + vector search)
- 📚 Add clinical guideline retrieval with chunk-level vector search
- 📥 Build a PubMed XML ingestion pipeline
- 🤖 Implement Agentic RAG (multi-round iterative retrieval)
- 👤 Add user system + query history

---

<div align="center">

**MIT License** · Made for learning and sharing

</div>