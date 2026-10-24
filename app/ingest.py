import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
        chunk_size=800, # 调大，让一段代码尽量在一整块里
        chunk_overlap=100, # 调大重叠，防止切断上下文
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""] # 尽量按语义边界切
    )

    chunks = text_splitter.split_documents(documents)

    # 【新增过滤逻辑】
    # 修改：放宽长度限制，并去掉写死的词
    clean_chunks = []
    for chunk in chunks:
        # 长度大于 30，并且不是纯靠符号堆砌的块
        text = chunk.page_content.strip()
        if len(text) > 30:
            clean_chunks.append(chunk)

    print(f"🧹 清洗后，剩余 {len(clean_chunks)} 个有效块。")
    return clean_chunks


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