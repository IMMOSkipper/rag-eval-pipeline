import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from app.generate import ask_rag # 导入写好的问答函数
from app.ingest import async_ingest_document
from app.celery_app import celery_app
from app.agent import async_run_agent
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

load_dotenv()
app = FastAPI(title="RAG Eval Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求体格式
class QueryRequest(BaseModel):
    query: str
    model_name: str = os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017") # 允许通过接口指定模型

class AgentRequest(BaseModel):
    query: str
    session_id: str = "test_user_1" # 新增

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = ask_rag(request.query, model_name=request.model_name)
        return {
            "code": 200,
            "data": {
                "query": request.query,
                "answer": result
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并异步解析入库"""
    # 确保目录存在
    os.makedirs("data/docs", exist_ok=True)
    file_path = f"data/docs/{file.filename}"

    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 丢给 Celery 异步处理
    task = async_ingest_document.delay(file_path)

    return {
        "code": 200,
        "message": "文件已上传，正在后台解析入库",
        "task_id": task.id
    }

@app.get("/api/task/{task_id}")
def get_task_status(task_id: str):
    """查询异步任务状态"""
    task = celery_app.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {"status": "PENDING", "message": "任务等待中或未找到"}
    elif task.state == 'SUCCESS':
        return {"status": "SUCCESS", "result": task.result}
    elif task.state == 'FAILURE':
        return {"status": "FAILURE", "error": str(task.info)}
    else:
        return {"status": task.state}

# 发起 Agent 任务
@app.post("/api/agent")
def run_agent_endpoint(request: AgentRequest):
    task = async_run_agent.delay(request.query, request.session_id)
    return {"code": 200, "task_id": task.id}

# SSE 实时推送状态流
@app.get("/api/agent/stream/{task_id}")
async def stream_agent_status(task_id: str):
    async def event_generator():
        last_meta = None
        while True:
            # 用 asyncio.to_thread 把同步阻塞的 Celery 查询放到后台线程
            # 这样即使有 1000 个并发，也不会卡死 FastAPI 的主线程
            task = await asyncio.to_thread(celery_app.AsyncResult, task_id)
            state = await asyncio.to_thread(lambda: task.state)

            # 如果状态变了，推送给前端
            # 【核心修改】判断具体的 info（内容）是否变化，而不是只看 state（大状态）
            if state == 'PROGRESS':
                info = await asyncio.to_thread(lambda: task.info)
                if info != last_meta:
                    last_meta = info
                    yield f"data: {json.dumps(info, ensure_ascii=False)}\n\n"
            elif state == 'SUCCESS':
                result = await asyncio.to_thread(lambda: task.result)
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                break
            elif state == 'FAILURE':
                yield f"data: {json.dumps({'status': 'failed'}, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(1) # 每秒轮询一次 Redis 状态

    return StreamingResponse(event_generator(), media_type="text/event-stream")