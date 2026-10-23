from fastapi import FastAPI

app = FastAPI(title="RAG Evaluation Pipeline", description="A pipeline for evaluating retrieval-augmented generation (RAG) models.", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}