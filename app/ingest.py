import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOC_PATH = "data/docs/langchain_overview.md"
DB_DIR = "chroma_db" # 向量库保存的本地目录

def load_and_chunk():
    if not os.path.exists(DOC_PATH):
        print(f"❌️ 找不到文件：{DOC_PATH}，请检查路径！")
        return

    print(f"📄 正在加载文档：{DOC_PATH} ...")
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()

    print(f"✅️ 文档加载成功，共有 {len(documents)} 页/节")

    # 设置分块策略（这是 RAG 的第一个坑，我们先采用固定长度+重叠窗口）
    # chunk_size：每块的字符数，chunk_overlap：相邻块之间的重叠字符数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # 先设 500 字符，对中文来说大概是 250-300 个字
        chunk_overlap=50, # 50 个字符的重叠，防止语义在边界被切断
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""] # 尽量按语义边界切
    )

    return text_splitter.split_documents(documents)

def main():
    print("🚀 开始加载文档并切块...")
    chunks = load_and_chunk()
    print(f"✂️ 切块完成，共切出 {len(chunks)} 个块。\n")

    print("🧠 正在加载 Embedding 模型（beg-small-zh-v1.5）...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("🗄 正在将向量写入 Chroma 数据库...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"🎉 入库完成！数据库保存在：{DB_DIR}")

if __name__ == "__main__":
    main()