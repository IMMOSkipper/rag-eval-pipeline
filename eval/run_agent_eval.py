import os
import sys
import json
import time
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
from app.agent import async_run_agent
from app.celery_app import celery_app

celery_app.conf.task_always_eager = True

AGENT_QUESTIONS = [
    {"query": "查一下北京的天气", "expected_tool": "get_weather", "category": "纯天气"},
    {"query": "上海现在多少度？", "expected_tool": "get_weather", "category": "纯天气"},
    {"query": "人工智能的第一次寒冬是由什么引发的？", "expected_tool": "search_knowledge_base", "category": "纯知识库"},
    {"query": "Transformer 模型是哪一年提出来的？", "expected_tool": "search_knowledge_base", "category": "纯知识库"},
    {"query": "查一下广州的天气，并告诉我 LangChain 运行框架包含哪些组件？", "expected_tool": "both", "category": "混合调度"},
    {"query": "深圳天气怎么样？另外，感知机是谁公布的？", "expected_tool": "both", "category": "混合调度"},
    {"query": "LangChain 源码里的 `create_agent` 底层是如何实现状态持久化的？", "expected_tool": "refuse", "category": "边界拒答"},
    {"query": "请对比 DeepSeek 和 GPT-4 在 MoE 架构上的具体参数量差异？", "expected_tool": "refuse", "category": "边界拒答"}
]

def run_agent_eval():
    print("🚀 开始 Agent 指标评测...")
    total = len(AGENT_QUESTIONS)

    tool_correct = 0
    step_valid = 0
    answer_from_tool = 0

    for i, item in enumerate(AGENT_QUESTIONS):
        q = item["query"]
        expected = item["expected_tool"]
        category = item.get("category", "未知")

        print(f"\n[{i+1}/{total}] [{category}] 测试问题: {q}")

        start_t = time.time()
        result = async_run_agent.apply(args=(q, f"test_session_{i}")).get()

        ans = result.get("answer", "")
        metrics = result.get("metrics", {})
        steps = metrics.get("step", 0)

        # 步数合规
        if steps <= 5:
            step_valid += 1
        else:
            print(f"  ❌️ 步数超限: {steps}")

        # 工具选择判断
        refuse_keywords = ["无法回答", "没有找到", "未检索到", "没有说明", "无法确认"]
        is_refuse = any(kw in ans for kw in refuse_keywords)

        if expected == "get_weather":
            if "气温" in ans or "天气" in ans or "度" in ans:
                tool_correct += 1
                answer_from_tool += 1
                print(f"  ✅️ 工具选择正确：调用了天气工具")
        elif expected == "search_knowledge_base":
            if ("莱特希尔" in ans or "报告" in ans) or ("2017" in ans or "二〇一七" in ans) or ("罗森布拉特" in ans):
                tool_correct += 1
                answer_from_tool += 1
                print(f"  ✅️ 工具选择正确：调用了知识库工具")
        elif expected == "both":
            if ("气温" in ans or "天气" in ans or "度" in ans) and ("组件" in ans or "感知机" in ans):
                tool_correct += 1
                answer_from_tool += 1
                print(f"  ✅️ 工具选择正确：混合调用了天气和知识库工具")
        elif expected == "refuse":
            if is_refuse:
                tool_correct += 1
                print(f"  ✅️ 成功拒答")

        print(f"  ⏱ 耗时: {time.time() - start_t:.2f}s | 步数: {steps}步")
        time.sleep(1)

    print("\n" + "="*40)
    print("🏆 Agent 执行指标评测报告")
    print("="*40)
    print(f"🎯 工具选择准确率: {(tool_correct/total)*100:.1f}%")
    print(f"👣 步数合规率: {(step_valid/total)*100:.1f}% (<=5步)")
    print(f"📝 答案事实率: {(answer_from_tool/total)*100:.1f}% (回答契合预期数据源)")

if __name__ == "__main__":
    run_agent_eval()