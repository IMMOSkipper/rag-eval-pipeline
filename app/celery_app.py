import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 连接本地 Docker 跑的 Redis
celery_app = Celery(
    "rag_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    # 强制加载任务模块
    include=["app.ingest", "app.agent"]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    # 【新增】允许 Celery 记录并更新任务的中间状态
    task_track_started=True,
)