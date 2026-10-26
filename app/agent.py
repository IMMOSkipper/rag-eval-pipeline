import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017")

# 定义工具（仅用于演示）
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    # 实际开发中这里会调用第三方API
    if "北京" in city:
        return "北京今天晴，气温25℃，适合户外运动。"
    return f"抱歉，没有查到 {city} 的天气数据。"

# 编写 Agent 的系统提示词
AGENT_PROMPT = """你是一个智能助手，可以调用以下工具来帮助用户解决问题：
1. get_weather(city: str): 查询指定城市的天气。

请严格遵循以下格式回答：
Question: 用户的问题
Thought: 思考应该使用哪个工具
Action: 工具名称，比如 get_weather
Action Input: 工具的输入参数，比如：北京

如果不需要使用工具，或者你已经得到了最终答案，请使用以下格式：
Thought: 我已经知道答案了
Final Answer: 最终的回答内容
"""

def run_agent_step_by_step(query: str, model_name=DEFAULT_MODEL):
    """手搓一个单步的 Agent，生成器模式，方便后续做 SSE 流式输出"""
    print(f"\n🙋‍♂️ 用户：{query}")

    # 初始的 Prompt
    prompt = f"{AGENT_PROMPT}\nQuestion: {query}\n"

    # 最多循环3次，防止死循环
    for step in range(3):
        print(f"⌛️ [Agent 思考中] 第 {step + 1} 轮...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stop=["\nObservation:"] # 遇到 Observation 就停止，等待工具执行
        )

        output = response.choices[0].message.content
        print(f"🧠 模型输出：\n{output}")

        # 解析大模型的输出
        if "Final Answer:" in output:
            final_answer = output.split("Final Answer:")[-1].strip()
            print(f"✅️ 最终回答：{final_answer}\n")
            return final_answer

        # 如果决定调用工具
        if "Action:" in output and "Action Input:" in output:
            try:
                action = output.split("Action:")[-1].split("Action Input:")[0].strip()
                action_input = output.split("Action Input:")[-1].strip()

                print(f"🛠️ 准备调用工具：{action}, 参数：{action_input}")

                # 执行工具
                if action == "get_weather":
                    observation = get_weather(action_input)
                else:
                    observation = f"找不到工具 {action}"

                print(f"👀 工具执行结果（Observation）：{observation}")

                # 把观察结果塞回 Prompt 里，让模型继续思考
                prompt += f"{output}\nObservation：{observation}\nThought:"

            except Exception as e:
                print(f"解析工具调用失败：{e}")
                break

    return "Agent 执行失败或超时。"

if __name__ == "__main__":
    run_agent_step_by_step("查一下北京天气怎么样？")