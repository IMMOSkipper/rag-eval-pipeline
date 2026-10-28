# 🚀 RAG Eval Pipeline: 大模型知识库问答与异步 Agent 调度服务

一个基于 FastAPI 构建的生产级大模型应用实战项目。本项目不仅实现了完整的 RAG（检索增强生成）链路，还针对**大模型推理耗时**、**库外问题幻觉**、**Agent 多步决策**以及**前端实时反馈**等企业落地痛点，提供了完整的工程化异步调度方案。

## ✨ 核心特性 (Features)

*   **高并发异步解耦**：引入 Celery + Redis 实现文档解析与向量化（Embedding）的异步任务队列，彻底解决大文件上传导致的接口阻塞问题，接口响应降至毫秒级。
*   **双重防幻觉机制**：
    *   前置拦截：基于向量检索距离阈值，直接拒答无关问题，节省 Token。
    *   后置约束：通过结构化 Prompt 严格限制大模型仅根据检索上下文作答。
*   **ReAct Agent 工作流**：手搓 Thought-Action-Observation 推理循环，实现智能体对工具（如外部 API）的调用。通过 SSE (Server-Sent Events) 将 Agent 思考过程实时流式推流至前端。
*   **异步流式 IO 优化**：在 SSE 接口中使用 asyncio.to_thread 将 Celery 同步查询卸载到线程池，完美解决高并发长连接场景下的事件循环（Event Loop）阻塞问题。
*   **自动化评测流水线**：内置 eval/ 评测脚本，包含正常问答与库外拒答题库。通过量化指标（准确率 83.3%）驱动检索参数与 Prompt 调优。
*   **多模型无缝适配**：遵循 12-Factor 原则，解耦 API Key 与 Base URL。封装统一的大模型调用客户端，支持在接口层参数化切换 DeepSeek、GPT 等多种模型。
*   **一键容器化部署**：提供完整的 Docker Compose 编排（API、Worker、Redis），并通过 Bind Mount 机制挂载本地模型缓存，实现容器秒级启动。

## 🛠️ 技术栈 (Tech Stack)

*   **后端框架**：Python 3.10+, FastAPI (SSE 流式输出, CORS 跨域支持)
*   **异步调度**：Celery, Redis (Broker & Backend)
*   **大模型与 RAG**：LangChain, ChromaDB (向量库), BAAI/bge-small-zh-v1.5 (Embedding), DeepSeek / OpenAI 兼容接口
*   **部署与工程化**：Docker, Docker Compose, python-dotenv

## 🚀 快速开始 (Quick Start)

### 1. 克隆项目与配置环境变量

    git clone <your-repo-url>
    cd rag-eval-pipeline
    cp .env.example .env

编辑 .env 文件，填入你的大模型 API 信息：

    OPENAI_API_KEY="sk-your-api-key"
    OPENAI_BASE_URL="https://api.your-provider.com/v1"
    DEFAULT_MODEL="deepseek-v4.1-flash"

### 2. 一键 Docker 部署 (推荐)

*(请确保宿主机已安装 Docker Desktop)*

    docker compose up --build -d

> **性能优化提示**：docker-compose.yml 中已将宿主机的 ~/.cache/huggingface 挂载到容器内。这样容器无需重新下载大模型权重，实现了秒级启动。

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

*   POST /api/upload：上传文档（异步）。返回 task_id。
*   GET /api/task/{task_id}：轮询查询文档解析与入库状态。
*   POST /api/chat：基于知识库的常规同步问答。可传入 model_name 切换模型。
*   POST /api/agent：发起 Agent 任务。返回 task_id。
*   GET /api/agent/stream/{task_id}：**SSE 接口**，实时推送 Agent 的思考与工具调用状态。

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
    │   ├── retrieve.py      # 向量检索与阈值拦截
    │   ├── generate.py      # 大模型 Prompt 拼装与生成
    │   └── agent.py         # 手搓 ReAct Agent 推理循环
    ├── eval/
    │   ├── questions.jsonl  # 自建评测集（含正常与拒答）
    │   └── run_eval.py      # 自动化评测脚本
    ├── data/docs/           # 测试文档存放目录
    ├── .env.example         # 环境变量模板
    ├── Dockerfile           # 镜像构建文件（含国内源优化）
    ├── docker-compose.yml   # API、Worker、Redis 容器编排
    ├── requirements.txt     # Python 依赖清单
    └── README.md