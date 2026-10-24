import os
from dotenv import load_dotenv
from openai import OpenAI
from app.retrieve import get_relevant_context

# 加载 .env 里的环境变量
load_dotenv()

# 初始化大模型客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

MODEL_NAME = "deepseek-v4.1-flash-1017"

def ask_rag(query: str):
    print(f"\n🙋‍♂️ 用户：{query}")

    # 检索上下文
    context = get_relevant_context(query)

    # 如果没检索到有效内容，直接拒答
    if not context:
        print("🤖模型：抱歉，知识库没有找到与您问题相关的内容，我无法回答。")
        return

    # 组装 Prompt
    prompt = f"""你是一个严谨的AI助手，请严格根据下面提供的参考资料来回答用户的问题。
    如果你在参考资料中找不到答案，请直接说“根据现有资料，我无法回答这个问题”，绝对不要编造。
    
    【参考资料】：
    {context}
    
    【用户问题】：
    {query}
    """

    # 调用大模型
    print("🤖模型正在思考中...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # 温度调低，让模型回答更确定
            timeout=30
        )
        answer = response.choices[0].message.content
        print(f"\n🤖 模型回答：\n{answer}\n")
        print("-" * 50)
    except Exception as e:
        print(f"调用大模型失败：{e}")

if __name__ == "__main__":
    # 测试1：知识库里有答案的问题
    ask_rag("怎么自定义工具给Agent使用？")

    # 测试2：知识库里没有的库外问题
    ask_rag("红烧肉怎么做好吃？")