import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"

def search(query: str, top_k: int = 2):
    print(f"🔍 用户提问：{query}")

    # 加载同样的 Embedding 模型（注意：要和入库时用的模型一模一样）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 加载本地已有的 Chroma 向量数据库
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # 相似度检索：找出最相关的 top_k 个块
    # 注意：在 Chroma 中，返回的 score 是“距离”（越小越相似）
    docs_with_score = vectorstore.similarity_search_with_score(query, k=top_k)

    print("-" * 40)
    for i, (doc, score) in enumerate(docs_with_score):
        print(f"【匹配结果 {i+1}】距离分数：{score:.4f}")
        # 截取前 150 个字符打印，方便看结果
        content_preview = doc.page_content.replace("\n", " ")[:150]
        print(f"内容预览：{content_preview}...\n")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    # 准备几个问题来测试
    # 问题1：文档里有的概念
    search("LangChain 里的智能体（Agent）是什么？")

    # 问题2：文档里有的代码相关
    search("怎么自定义工具给 Agent 使用？")

    # 问题3：故意问一个文档里没有的内容（测试会不会瞎找）
    search("红烧肉怎么做好吃？")