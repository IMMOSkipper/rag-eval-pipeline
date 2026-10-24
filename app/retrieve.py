import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = "chroma_db"
THRESHOLD = 0.9 # 调大，避免误杀

# 全局加载模型，避免每次调用都重新加载
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def get_relevant_context(query: str, top_k: int = 3):
    """检索相关文档并拼接成上下文"""
    docs_with_score = vectorstore.similarity_search_with_score(query, k=top_k)

    valid_contexts = []
    print(f"\n🔍 [检索阶段] 用户提问：{query}")

    for i, (doc, score) in enumerate(docs_with_score):
        if score <= THRESHOLD:
            valid_contexts.append(doc.page_content)
            print(f" ✅️ 命中块 {i+1} (距离 {score:.4f}): {doc.page_content[:50]}...")
        else:
            print(f" ❌️ 拒绝块 {i+1} (距离 {score:.4f} 超阈值)")

    if not valid_contexts:
        return ""

    # 将多个块用分隔符拼起来作为上下文
    return "\n\n---\n\n".join(valid_contexts)

if __name__ == "__main__":
    # 准备几个问题来测试
    # 问题1：文档里有的概念
    search("LangChain 里的智能体（Agent）是什么？")

    # 问题2：文档里有的代码相关
    search("怎么自定义工具给 Agent 使用？")

    # 问题3：故意问一个文档里没有的内容（测试会不会瞎找）
    search("红烧肉怎么做好吃？")