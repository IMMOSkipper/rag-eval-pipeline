import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from app.generate import ask_rag # 导入写好的问答函数
from app.ingest import async_ingest_document
from app.celery_app import celery_app

load_dotenv()
app = FastAPI(title="RAG Eval Pipeline API")

# 定义请求体格式
class QueryRequest(BaseModel):
    query: str
    model_name: str = os.getenv("DEFAULT_MODEL") # 允许通过接口指定模型

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