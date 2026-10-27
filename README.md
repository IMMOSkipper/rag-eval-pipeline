# 🚀 RAG Eval Pipeline: 大模型知识库问答与自动化评测服务

一个基于 FastAPI 构建的生产级大模型应用实战项目。本项目不仅实现了完整的 RAG（检索增强生成）问答链路，还针对**大模型推理耗时**、**库外问题幻觉**以及**模型效果评估难**这三大企业落地痛点，提供了工程化的解决方案。

## ✨ 核心特性 (Features)

*   **高并发异步解析**：引入 `Celery + Redis` 实现文档解析与向量化（Embedding）的异步任务队列，彻底解决大文件上传导致的接口阻塞问题。
*   **双重防幻觉机制**：
    *   前置拦截：基于 Chroma 向量检索距离阈值，直接拒答无关问题，节省 Token。
    *   后置约束：通过结构化 Prompt 严格限制大模型仅根据检索上下文作答。
*   **自动化评测流水线**：内置 `eval/` 评测脚本，包含正常问答与库外拒答题库。通过量化指标（准确率 83.3%）驱动检索参数与 Prompt 调优。
*   **多模型无缝适配**：遵循 12-Factor 原则，解耦底层鉴权配置，支持在请求参数中无缝切换 DeepSeek、GPT 等不同大模型。
*   **Agent 推理与流式输出**：手搓基于 ReAct 模式的 Agent 决策循环，集成 SSE 实现思考状态的实时流式推送。

## 🛠️ 技术栈 (Tech Stack)

*   **后端框架**：Python 3.10+, FastAPI
*   **异步调度**：Celery, Redis
*   **大模型与 RAG**：LangChain, ChromaDB (向量库), BAAI/bge-small-zh-v1.5 (Embedding 模型)
*   **部署与配置**：Docker, python-dotenv

## 🚀 快速开始 (Quick Start)

### 1. 克隆项目与安装依赖
    git clone <your-repo-url>
    cd rag-eval-pipeline
    python -m venv .venv
    source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
    pip install -r requirements.txt

### 2. 配置环境变量
复制 `.env.example` 为 `.env`，并填入你的大模型 API 信息（兼容 OpenAI 接口格式的均可）：
    OPENAI_API_KEY="sk-your-api-key"
    OPENAI_BASE_URL="https://api.your-provider.com/v1"
    DEFAULT_MODEL="deepseek-chat"

### 3. 启动 Redis 服务 (使用 Docker)
    docker run -d --name redis-rag -p 6379:6379 redis

### 4. 启动后端服务 (需打开两个终端)

**终端 1：启动 Celery 异步 Worker**
    celery -A app.ingest worker --loglevel=info

**终端 2：启动 FastAPI 服务**
    uvicorn app.main:app --reload

访问 `http://127.0.0.1:8000/docs` 即可查看并测试 API 接口。

## 📡 API 接口说明

*   `POST /api/upload`：上传文档（异步）。返回 `task_id`。
*   `GET /api/task/{task_id}`：轮询查询异步解析状态。
*   `POST /api/chat`：基于知识库进行问答。可传入 `model_name` 切换模型。

## 🧪 自动化评测 (Evaluation)

本项目自带了一套轻量级的自动化评测流水线，用于验证系统的检索质量与防幻觉能力。

运行评测脚本：
    python eval/run_eval.py

*评测结果将输出每道题目的检索命中情况、模型回答以及最终的通过率。*

## 📂 项目结构

    rag-eval-pipeline/
    ├── app/
    │   ├── main.py          # FastAPI 入口与路由
    │   ├── celery_app.py    # Celery 实例配置
    │   ├── ingest.py        # 文档异步解析与向量化入库
    │   ├── retrieve.py      # 向量检索与阈值拦截
    │   ├── generate.py      # 大模型生成与 Prompt 拼装
    │   └── agent.py         # (To be updated) Agent 工作流
    ├── eval/
    │   ├── questions.jsonl  # 自建评测集
    │   └── run_eval.py      # 自动化评测脚本
    ├── data/docs/           # 测试文档存放目录
    ├── .env.example         # 环境变量模板
    ├── requirements.txt     # 依赖清单
    └── README.md