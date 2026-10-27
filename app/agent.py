import os
from dotenv import load_dotenv
from openai import OpenAI
from app.celery_app import celery_app # 引入 Celery
# import time

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
    # time.sleep(15)
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

@celery_app.task(bind=True)
def async_run_agent(self, query: str, model_name=DEFAULT_MODEL):
    prompt = f"{AGENT_PROMPT}\nQuestion: {query}\n"

    for step in range(3):
        # 【关键】每次思考前，通过 Celery 更新中间状态，前端就轮询到了
        # status_info = f"Step {step + 1}: 模型思考中..."
        self.update_state(state="PROGRESS", meta={'step': step + 1, 'status': '模型思考中...'})
        # print(f"🟢 [Celery 状态更新] {status_info}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stop=["\nObservation:"] # 遇到 Observation 就停止，等待工具执行
        )

        output = response.choices[0].message.content

        if "Final Answer:" in output:
            final_answer = output.split("Final Answer:")[-1].strip()
            return {"status": "success", "answer": final_answer}

        if "Action:" in output and "Action Input:" in output:
            action = output.split("Action:")[-1].split("Action Input:")[0].strip()
            action_input = output.split("Action Input:")[-1].strip()

            # 【关键】更新状态为执行工具
            self.update_state(state='PROGRESS', meta={'step': step + 1, 'status': f'正在调用工具: {action}'})
            # print(f"🛠️ [Celery 状态更新] 正在调用工具: {action}")

            if action == "get_weather":
                observation = get_weather(action_input)
            else:
                observation = f"找不到工具 {action}"

            prompt += f"{output}\nObservation：{observation}\nThought:"


    return {"status": "failed", "answer": "Agent 执行超时"}

if __name__ == "__main__":
    run_agent_step_by_step.delay("查一下北京天气怎么样？")