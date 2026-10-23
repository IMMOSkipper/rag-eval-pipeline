import os
# 修改：把 UnstructuredMarkdownLoader 替换成 TextLoader 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 指定文档路径（相对项目根目录）
DOC_PATH = "data/docs/langchain_overview.md"

def load_and_chunk():
    if not os.path.exists(DOC_PATH):
        print(f"❌️ 找不到文件：{DOC_PATH}，请检查路径！")
        return

    print(f"📄 正在加载文档：{DOC_PATH} ...")
    # 加载 Markdown 文档 
    # 修改：使用 TextLoader，并指定编码为 utf-8
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

    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 切块完成，共切出 {len(chunks)} 个块。\n")

    # 打印前3个块看看长什么样
    print("-" * 40)
    for i, chunk in enumerate(chunks[:3]):
        print(f"【块 {i+1}】长度：{len(chunk.page_content)} 字符")
        print(chunk.page_content)
        print("-" * 40)

if __name__ == "__main__":
    load_and_chunk()