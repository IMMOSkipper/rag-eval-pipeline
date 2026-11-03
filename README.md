# 🚀 RAG Eval Pipeline: 大模型知识库问答与异步 Agent 调度服务

一个基于 FastAPI 构建的大模型 RAG 与 Agent 实战项目，聚焦长耗时任务的异步调度与工程化落地。本项目不仅实现了完整的 RAG（检索增强生成）链路，还针对**大模型推理耗时**、**库外问题幻觉**、**Agent 多步决策**以及**前端实时反馈**等企业落地痛点，提供了完整的工程化异步调度方案。

## ✨ 核心特性 (Features)

*   **高并发异步解耦**：引入 `Celery + Redis` 实现文档解析与向量化（Embedding）的异步任务队列，彻底解决大文件上传导致的接口阻塞问题，接口响应降至毫秒级。
*   **模块化的 RAG 链路与 Rerank 重排**：集成 LangChain + ChromaDB + BGE 向量模型，实现文档切块与检索。引入 `CrossEncoder` 进行两阶段检索（粗排+精排），结合阈值截断与 Prompt 约束，有效降低库外幻觉。
*   **标准 ReAct Agent 工作流**：手搓基于标准 `Function Calling` (JSON Schema) 的 Agent 决策循环，支持多轮对话（Redis 短期记忆）。并通过 SSE (Server-Sent Events) 将 Agent 思考过程实时推流至前端。
*   **双重熔断防死循环**：Agent 内置 `Max Steps (最大步数)` 与 `Max Execution Time (最大执行时间)` 双重熔断机制，彻底杜绝逻辑死循环与外部 API 卡死导致的 Token 浪费与线程阻塞。
*   **异步流式 IO 优化**：在 SSE 接口中使用 `asyncio.to_thread` 将 Celery 同步查询卸载到线程池，完美解决高并发长连接场景下的事件循环（Event Loop）阻塞问题。
*   **自动化评测流水线**：内置 `eval/` 评测脚本，包含正常问答与库外拒答题库。通过量化指标（内置一套 6 题的自建评测集（通过率 5/6），用于验证检索与拒答逻辑）驱动检索参数与 Prompt 调优。
*   **多模型无缝适配**：遵循 12-Factor 原则，解耦 API Key 与 Base URL。封装统一的大模型调用客户端，支持在接口层参数化切换 DeepSeek、GPT 等多种模型。
*   **一键容器化与极速启动**：提供完整的 Docker Compose 编排（API、Worker、Redis），并通过 Bind Mount 机制挂载本地模型缓存与代码目录，实现容器秒级启动与代码热更新。

## 🛠️ 技术栈 (Tech Stack)

*   **后端框架**：Python 3.10+, FastAPI (SSE 流式输出, CORS 跨域支持)
*   **异步调度**：Celery, Redis (Broker & Backend)
*   **大模型与 RAG**：LangChain, ChromaDB (向量库), BAAI/bge-small-zh-v1.5 (Embedding), BAAI/bge-reranker-base (Rerank), DeepSeek / OpenAI 兼容接口
*   **部署与工程化**：Docker, Docker Compose, python-dotenv

## 🚀 快速开始 (Quick Start)

### 1. 克隆项目与配置环境变量

    git clone <your-repo-url>
    cd rag-eval-pipeline
    cp .env.example .env

编辑 .env 文件，填入你的大模型 API 信息：

    OPENAI_API_KEY="sk-your-api-key"
    OPENAI_BASE_URL="https://api.your-provider.com/v1"
    DEFAULT_MODEL="deepseek-chat"

### 2. 一键 Docker 部署 (推荐)

*(请确保宿主机已安装 Docker Desktop)*

    docker compose up -d --build

> **性能与空间优化提示**：docker-compose.yml 中已将宿主机的 ~/.cache/huggingface 和当前代码目录挂载到容器内。容器无需重新下载大模型权重，且修改 Python 代码后只需 `docker compose restart` 即可秒级生效，绝不占用额外磁盘。

### 3. 本地开发模式 (可选)

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

    # 启动 Redis
    docker run -d --name redis-rag -p 6379:6379 redis

    # 终端 1: 启动 FastAPI
    uvicorn app.main:app --reload

    # 终端 2: 启动 Celery Worker
    celery -A app.celery_app worker --loglevel=info

访问 http://127.0.0.1:8000/docs 即可查看并测试 API 接口。

## 📡 API 接口说明

*   `POST /api/upload`：上传文档（异步）。返回 `task_id`。
*   `GET /api/task/{task_id}`：轮询查询文档解析与入库状态。
*   `POST /api/chat`：基于知识库的常规同步问答。可传入 `model_name` 切换模型。
*   `POST /api/agent`：发起 Agent 任务。可传入 `query` 和 `session_id`。返回 `task_id`。
*   `GET /api/agent/stream/{task_id}`：**SSE 接口**，实时推送 Agent 的思考与工具调用状态。

## 🧪 自动化评测 (Evaluation)

本项目自带了一套轻量级的自动化评测流水线，用于验证系统的检索质量与防幻觉能力。

运行评测脚本：

    python eval/run_eval.py

*评测结果将输出每道题目的检索命中情况、模型回答以及最终的通过率（基准准确率 83.3%）。*

## 📂 项目结构

    rag-eval-pipeline/
    ├── app/
    │   ├── main.py          # FastAPI 入口、路由、CORS 与 SSE 流式响应
    │   ├── celery_app.py    # Celery 实例与环境变量读取
    │   ├── ingest.py        # 文档异步解析与向量化入库
    │   ├── retrieve.py      # 向量检索、阈值拦截与 Rerank 精排
    │   ├── generate.py      # 大模型 Prompt 拼装与生成
    │   └── agent.py         # 标准 Function Calling 与 ReAct 循环控制
    ├── eval/
    │   ├── questions.jsonl  # 自建评测集（含正常与拒答）
    │   └── run_eval.py      # 自动化评测脚本
    ├── data/docs/           # 测试文档存放目录
    ├── .env.example         # 环境变量模板
    ├── Dockerfile           # 镜像构建文件（含国内源优化）
    ├── docker-compose.yml   # API、Worker、Redis 容器编排
    ├── requirements.txt     # Python 依赖清单
    └── README.md