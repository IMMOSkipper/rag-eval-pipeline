import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from app.celery_app import celery_app # 引入 Celery
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017")

# 连接 Redis 用于存储短期记忆（Local 用 localhost，Docker 用 redis）
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# 定义工具（加入 Tenacity 容错重试）
@retry(
    stop=stop_after_attempt(3), # 最多重试3次
    wait=wait_exponential(multiplier=1, min=1, max=5), # 指数退避重试
    retry=retry_if_exception_type(Exception)
)
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    print(f"🛠️ [工具调用] 正常查询 {city} 的天气...")
    # 模拟可能发生的网络错误（可以试着打开这行看看重试效果）
    # if time.time() % 2 == 0: raise Exception("Network Timeout!")
    # 实际开发中这里会调用第三方API
    # time.sleep(15)
    if "北京" in city:
        return "北京今天晴，气温25℃，适合户外运动。"
    return f"抱歉，没有查到 {city} 的天气数据。"

# 把 Python 函数映射成字典，方便以后根据名字调用
AVAILABLE_TOOLS = {
    "get_weather": get_weather
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

@celery_app.task(bind=True)
def async_run_agent(self, query: str, session_id: str = "default_user", model_name=DEFAULT_MODEL):
    # 从 Redis 取出历史记忆
    history_key = f"chat_history:{session_id}"
    history_json = r.get(history_key)
    messages = json.loads(history_json) if history_json else []

    # 如果没有 system prompt，就加一个
    if not messages:
        messages.append({
            "role": "system",
            "content": "你是一个智能助手，可以调用工具来帮助用户解决问题。"
        })

    # 把当前用户的问题加入记忆
    messages.append({"role": "user", "content": query})

    # 【新增】记录开始时间，防止由于 API 卡死导致任务一直挂起
    start_time = time.time()
    max_execution_time = 60 # 最对允许执行60秒

    # Agent 思考循环（最多5步防止死循环）
    for step in range(5):
        # 【新增】超时熔断拦截
        if time.time() - start_time > max_execution_time:
            print("⚠️ Agent 执行超时，强制中断！")
            return {"status": "failed", "answer": "任务执行时间过长，已自动中断。"}
        # 【关键】每次思考前，通过 Celery 更新中间状态，前端就轮询到了
        # status_info = f"Step {step + 1}: 模型思考中..."
        self.update_state(state="PROGRESS", meta={'step': step + 1, 'status': '模型思考中...'})
        # print(f"🟢 [Celery 状态更新] {status_info}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_SCHEMA,
            temperature=0.1,
        )

        response_message = response.choices[0].message
        # 把模型这轮的回复加入上下文；注意：必须转字典，否则存Redis会报错
        messages.append(response_message.dict())

        # 判断是否需要调用工具
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            self.update_state(state='PROGRESS', meta={'step': step + 1, 'status': f'正在调用工具: {function_name}'})
            print(f"🧠 模型决定调用工具: {function_name}, 参数: {function_args}")

            # 执行工具
            if function_name in AVAILABLE_TOOLS:
                try:
                    function_response = AVAILABLE_TOOLS[function_name](**function_args)
                except Exception as e:
                    # 如果工具报错（比如重试3次全失败），把错误信息作为观察结果塞回去
                    function_response = f"工具执行失败: {str(e)}"
            else:
                function_response = f"找不到工具 {function_name}"

            # 把工具执行结果加入上下文（注意 role 是 tool）
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_response)
            })
        else:
            # 没有工具调用，说明思考结束，输出了最终答案
            final_answer = response_message.content
            # 保存最新的记忆到 Redis（保留最近10条消息，设置1小时过期）
            r.setex(history_key, 3600, json.dumps(messages[-10:]))
            return {"status": "success", "answer": final_answer}

    return {"status": "failed", "answer": "Agent 执行步骤超出限制，已强制中断。"}