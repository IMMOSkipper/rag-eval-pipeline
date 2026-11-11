import os
import json
import time
import requests
import urllib.parse
from dotenv import load_dotenv
from openai import OpenAI
from app.celery_app import celery_app # 引入 Celery
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, ValidationError
from app.retrieve import get_relevant_context

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017")

# 连接 Redis 用于存储短期记忆（Local 用 localhost，Docker 用 redis）
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# 定义工具的入参模型
class WeatherArgs(BaseModel):
    city: str

# 知识库工具的参数
class KnowledgeArgs(BaseModel):
    query: str

# 定义一个异常，专门用来触发重试
class RetryableNetworkError(Exception):
    pass

# 定义工具（加入 Tenacity 容错重试，只对网络错误重试）
@retry(
    stop=stop_after_attempt(3), # 最多重试3次
    wait=wait_exponential(multiplier=1, min=1, max=5), # 指数退避重试
    retry=retry_if_exception_type(RetryableNetworkError)
)
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    print(f"🛠️ [真实 API 调用] 正常查询 {city} 的天气...")

    # time.sleep(35)
    # 使用免费的 wttr.in API，返回 JSON 格式
    # URL编码防御
    encoded_city = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded_city}?format=j1"

    try:
        # 真实的网络请求，强制设置3秒超时，防止卡死
        response = requests.get(url, timeout=3)

        # 遇到限流或服务端错误，触发重试
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableNetworkError(f"服务器繁忙，状态码：{response.status_code}")

        # 客户端错误（比如404），直接抛普通Exception，不触发重试，交给模型兜底
        if response.status_code == 404:
            raise Exception(f"找不到城市 {city} 的天气数据，请检查城市名。")

        data = response.json()

        # 解析失败，也不触发重试，交给模型
        try:
            temp = data['current_condition'][0]['temp_C']
            return f"{city} 当前气温 {temp}℃，天气数据获取成功。"
        except (KeyError, IndexError):
            raise Exception(f"无法解析 {city} 的天气数据，请检查城市名称。")  
    except requests.exceptions.RequestException as e:
        # 网络级别的超时或断开，才触发重试
        raise RetryableNetworkError(f"网络请求超时或断开：{str(e)}")
    
def search_knowledge_base(query: str) -> str:
    """从本地知识库中检索相关信息"""
    print(f"📚 [工具调用] 正在检索知识库: {query}")
    context = get_relevant_context(query, use_rerank=True)
    if not context:
        print("知识库中没有找到相关信息。")
    return context


# 把 Python 函数映射成字典，方便以后根据名字调用
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "当用户询问关于LangChain、大模型技术、文档内容等内部知识时，使用此工具检索知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的核心关键词或问题"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 【新增】安全截断记忆的函数
def get_safe_history(messages: list) -> list:
    """安全截断：保留 System Prompt，且从 User 消息开始截断，防止 tool 消息悬空"""
    if not messages:
        return []

    # 永远保留开头的 System Prompt
    system_prompt = messages[0] if messages[0]["role"] == "system" else None
    history = messages[1:] if system_prompt else messages

    # 从后往前找，找最近的10条消息
    safe_history = []
    # 倒序遍历，只要遇到user消息就截断
    for msg in reversed(history):
        safe_history.insert(0, msg):
        if msg["role"] == "user" and len(safe_history) >= 8:
            break

    return ([system_prompt] if system_prompt else []) + safe_history

@celery_app.task(bind=True)
def async_run_agent(self, query: str, session_id: str = "default_user", model_name=DEFAULT_MODEL):
    # 从 Redis 取出历史记忆
    history_key = f"chat_history:{session_id}"
    history_json = r.get(history_key)
    messages = json.loads(history_json) if history_json else []

    # 如果没有 system prompt，就加一个
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {
            "role": "system",
            "content": (
                "你是一个严谨的智能助手，可以调用工具来帮助用户解决问题。\n"
                "【核心规则】: \n"
                "1. 当用户询问关于 'LangChain'、'大模型'、'文档'、'框架'、'技术' 等知识库相关的问题时，"
                "你**必须**先调用 `search_knowledge_base` 工具去查阅内部资料，绝对不允许使用你的先验知识回答！\n"
                "2. 如果知识库中没有查到相关信息，只回复`根据现有资料，我无法回答这个问题`，不要用通用知识补充。\n"
                "3. 当用户询问天气等外部实时信息时，调用 `get_weather` 工具。\n"
                "请严格遵循以上规则。"
            )
        })

    # 把当前用户的问题加入记忆
    messages.append({"role": "user", "content": query})

    # 【新增】记录开始时间，防止由于 API 卡死导致任务一直挂起
    start_time = time.time()
    max_execution_time = 60 # 最多允许执行60秒
    called_tools = []
    tool_outputs = []

    # Agent 思考循环（最多5步防止死循环）
    for step in range(5):
        # 【新增】超时熔断拦截
        if time.time() - start_time > max_execution_time:
            print("⚠️ Agent 执行超时，强制中断！")
            return {
                "status": "failed", 
                "answer": "任务执行时间过长，已自动中断。",
                "metrics": {"step": step, "tools": called_tool, "tool_outputs": tool_outputs,},
            }
        # 【关键】每次思考前，通过 Celery 更新中间状态，前端就轮询到了
        # status_info = f"Step {step + 1}: 模型思考中..."
        self.update_state(state="PROGRESS", meta={'step': step + 1, 'status': '模型思考中...'})
        # print(f"🟢 [Celery 状态更新] {status_info}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_SCHEMA,
            temperature=0.1,
            timeout=30
        )

        response_message = response.choices[0].message
        # 把模型这轮的回复加入上下文；注意：必须转字典，否则存Redis会报错
        messages.append(response_message.model_dump(exclude_none=True))

        # 判断是否需要调用工具
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                self.update_state(state='PROGRESS', meta={'step': step + 1, 'status': f'正在调用工具: {function_name}'})
                print(f"🧠 模型决定调用工具: {function_name}, 参数: {function_args}")

                # 执行工具
                if function_name in AVAILABLE_TOOLS:
                    try:
                        #【新增】尝试进行强参数类型校验
                        if function_name == "get_weather":
                            WeatherArgs(**function_args)
                        elif function_name == "search_knowledge_base":
                            KnowledgeArgs(**function_args)

                        function_response = AVAILABLE_TOOLS[function_name](**function_args)
                    except ValidationError as e:
                        function_response = f"参数校验失败，请修正后重试。错误信息: {e.errors()}"
                        print(f"⚠️ 参数校验失败: {e}")
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

                called_tools.append(function_name)
                tool_outputs.append(str(function_response))
        else:
            # 没有工具调用，说明思考结束，输出了最终答案
            final_answer = response_message.content
            # 保存最新的记忆到 Redis（保留最近10条消息，设置1小时过期）
            safe_msgs = get_safe_history(messages)
            r.setex(history_key, 3600, json.dumps(safe_msgs))
            return {
                "status": "success", 
                "answer": final_answer, 
                "metrics": {"step": step + 1, "tools": called_tools, "tool_outputs": tool_outputs,},
                }

    return {
        "status": "failed",
        "answer": "Agent 执行步骤超出限制，已强制中断。",
        "metrics": {"step": 5, "tools": called_tools, "tool_outputs": tool_outputs,},
    }