import os
import sys
import json
import time
import re
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
from app.agent import async_run_agent
from app.celery_app import celery_app

celery_app.conf.task_always_eager = True

AGENT_QUESTIONS = [
    {"query": "查一下北京的天气", "expected_tool": "get_weather", "category": "纯天气"},
    {"query": "上海现在多少度？", "expected_tool": "get_weather", "category": "纯天气"},
    {"query": "人工智能的第一次寒冬是由什么引发的？", "expected_tool": "search_knowledge_base", "category": "纯知识库", "must_include": ["莱特希尔"]},
    {"query": "Transformer 模型是哪一年提出来的？", "expected_tool": "search_knowledge_base", "category": "纯知识库", "must_include_any": ["2017", "二〇一七"]},
    {"query": "查一下广州的天气，并告诉我 LangChain 运行框架包含哪些组件？", "expected_tool": "both", "category": "混合调度", "must_include": ["Prompt", "工具", "中间件"]},
    {"query": "深圳天气怎么样？另外，感知机是谁公布的？", "expected_tool": "both", "category": "混合调度", "must_include": ["罗森布拉特"]},
    {"query": "LangChain 源码里的 `create_agent` 底层是如何实现状态持久化的？", "expected_tool": "refuse", "category": "边界拒答"},
    {"query": "请对比 DeepSeek 和 GPT-4 在 MoE 架构上的具体参数量差异？", "expected_tool": "refuse", "category": "边界拒答"}
]

def run_agent_eval():
    print("🚀 开始 Agent 指标评测...")
    total = len(AGENT_QUESTIONS)

    tool_correct = 0
    step_valid = 0
    answer_from_tool = 0
    answer_total = 0
    refuse_correct = 0
    refuse_total = 0

    for i, item in enumerate(AGENT_QUESTIONS):
        q = item["query"]
        expected = item["expected_tool"]
        category = item.get("category", "未知")

        print(f"\n[{i+1}/{total}] [{category}] 测试问题: {q}")

        start_t = time.time()
        result = async_run_agent.apply(args=(q, f"test_session_{i}")).get()

        ans = result.get("answer", "")
        metrics = result.get("metrics") or {}
        if "tools" not in metrics or metrics.get("step") is None:
            print("  ❌️ 运行失败，没有返回 metrics")
            print(f"  ⏱ 耗时: {time.time() - start_t:.2f}s")
            if expected == "refuse":
                refuse_total += 1
            else:
                answer_total += 1
            continue

        steps = metrics["step"]
        tools = set(metrics["tools"])
        outputs = "\n".join(metrics.get("tool_outputs", []))

        if steps <= 5:
            step_valid += 1
        else:
            print(f"  ❌️ 步数超限: {steps}")

        if expected == "get_weather":
            tool_ok = tools == {"get_weather"}
        elif expected == "search_knowledge_base":
            tool_ok = tools == {"search_knowledge_base"}
        elif expected == "both":
            tool_ok = tools == {"get_weather", "search_knowledge_base"}
        else:
            tool_ok = tools == {"search_knowledge_base"}

        if tool_ok:
            tool_correct += 1
            print(f"  ✅️ 工具选择正确: {sorted(tools)}")
        else:
            print(f"  ❌️ 工具选择错误: {sorted(tools)}")

        if expected == "refuse":
            refuse_total += 1
            if "无法回答" in ans and tool_ok:
                refuse_correct += 1
                print("  ✅️ 拒答成功")
            else:
                print("  ❌️ 拒答失败")
        else:
            answer_total += 1
            temps = re.findall(r"当前气温\s*(-?\d+)", outputs)
            weather_ok = bool(temps) and all(t in ans for t in temps)
            facts = item.get("must_include", [])
            any_facts = item.get("must_include_any", [])
            fact_in_answer = all(x in ans for x in facts) and (not any_facts or any(x in ans for x in any_facts))
            fact_in_tool = all(x in outputs for x in facts) and (not any_facts or any(x in outputs for x in any_facts))
            fact_ok = fact_in_answer and fact_in_tool
            if expected == "get_weather":
                answer_ok = weather_ok
            elif expected == "search_knowledge_base":
                answer_ok = fact_ok
            else:
                answer_ok = weather_ok and fact_ok
            if answer_ok:
                answer_from_tool += 1
                print("  ✅️ 回答含工具结果")
            else:
                print("  ❌️ 回答未含工具结果")

        print(f"  ⏱ 耗时: {time.time() - start_t:.2f}s | 步数: {steps}步")
        time.sleep(1)

    print("\n" + "="*40)
    print("🏆 Agent 执行指标评测报告")
    print("="*40)
    print(f"🎯 工具选择准确率: {(tool_correct/total)*100:.1f}%")
    print(f"👣 步数合规率: {(step_valid/total)*100:.1f}% (<=5步)")
    print(f"📝 答案事实率: {(answer_from_tool/answer_total)*100:.1f}% (回答契合预期数据源)" if answer_total else "📝 答案事实率: 无有效题")
    print(f"🛡️ 拒答准确率: {refuse_correct}/{refuse_total}")

if __name__ == "__main__":
    run_agent_eval()