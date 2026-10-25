import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.celery_app import celery_app

# DOC_PATH = "data/docs/langchain_overview.md"
DB_DIR = "chroma_db" # 向量库保存的本地目录

def get_embeddings():
    """统一定义 Embedding 模型，方便复用"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def load_and_chunk(file_path: str):
    """核心逻辑：加载并切块"""
    if not os.path.exists(file_path):
        print(f"❌️ 找不到文件：{file_path}，请检查路径！")
        return

    print(f"📄 正在加载文档：{file_path} ...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print(f"✅️ 文档加载成功，共有 {len(documents)} 页/节")

    # 设置分块策略（这是 RAG 的第一个坑，我们先采用固定长度+重叠窗口）
    # chunk_size：每块的字符数，chunk_overlap：相邻块之间的重叠字符数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # 调大，让一段代码尽量在一整块里
        chunk_overlap=100, # 调大重叠，防止切断上下文
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""] # 尽量按语义边界切
    )

    chunks = text_splitter.split_documents(documents)

    # 清洗【新增过滤逻辑】
    # 修改：放宽长度限制，并去掉写死的词
    clean_chunks = []
    for chunk in chunks:
        # 长度大于 30，并且不是纯靠符号堆砌的块
        text = chunk.page_content.strip()
        if len(text) > 30:
            clean_chunks.append(chunk)

    print(f"🧹 清洗后，剩余 {len(clean_chunks)} 个有效块。")
    return clean_chunks

# ===============================
# Celery 异步任务入口（供 API 调用）
# ===============================
@celery_app.task
def async_ingest_document(file_path: str):
    """Celery 异步任务：读取、切块、向量化入库"""
    if not os.path.exists(file_path):
        return {"status": "error", "message": "文件不存在"}

    chunks = load_and_chunk(file_path)
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    return {"status": "success", "chunk_count": len(chunks)}


if __name__ == "__main__":
    TEST_DOC = "data/docs/langchain_overview.md"
    print(f"🧪 [本地测试] 开始同步入库：{TEST_DOC}")
    chunks = load_and_chunk(TEST_DOC)
    embeddings = get_embeddings()
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    print(f"✅️ [本地测试] 入库完成，共 {len(chunks)} 个块。")