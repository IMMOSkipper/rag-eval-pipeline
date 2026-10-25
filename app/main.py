import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app.generate import ask_rag # 导入写好的问答函数

load_dotenv()

app = FastAPI(title="RAG Eval Pipeline API", description="A pipeline for evaluating retrieval-augmented generation (RAG) models.", version="1.0.0")

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