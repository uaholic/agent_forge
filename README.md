# agent_forge

> AI Core Learning

一个记录自己学习 AI 的工程化实践项目。

目前主要围绕：

- LangChain
- LangGraph
- Milvus
- Embedding
- RAG
- Structured Output
- Agent Workflow

进行一些基础封装和实践。

项目目标不是造新的框架，而是通过不断拆解和实现，
理解 AI 应用工程中的核心模块。

---

# Features

当前已实现：

- Chat Chain Builder
- Structured Output Parser
- BGE Embedding 封装
- Milvus Hybrid Search
- 基础 RAG Pipeline
- 简单工程化目录结构

---

# Project Structure

```text
src/
├── embeddings/
│   └── bge.py
├── llm/
│   └── chat_chain.py
├── vectorstores/
│   └── milvus.py

examples/
├── chat_chain_demo.py
└── legal_rag_demo.py
```

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Environment Variables

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_base_url
```

---

# Examples

## LLM Demo
> examples/chat_chain_demo.py


## RAG Demo
> python examples/legal_rag_demo.py

---

# Current Tech Stack

- LangChain
- Milvus
- FlagEmbedding
- Pydantic
- Python 3.12

---

# Notes

这个项目主要用于个人学习记录。

代码会随着学习过程持续调整，
可能会存在：

- API 频繁变化
- 不稳定封装
- 实验性实现

但会尽量保持工程结构清晰。

---

# Future Plans

后续可能会继续尝试：

- LangGraph
- Memory
- Tool Calling
- Agent Runtime
- Multi Agent
- Workflow Engine
- MCP
- Streaming
- Async Pipeline
# Model Download

本项目默认不会提交本地模型文件。

请自行下载模型并放到以下目录：

```text
assets/models/
├── bge-base-zh-v1.5/
└── bge-m3/
```

---

## BGE Base

下载地址：

https://huggingface.co/BAAI/bge-base-zh-v1.5

---

## BGE M3

下载地址：

https://huggingface.co/BAAI/bge-m3