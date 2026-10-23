LangChain 中文教程V1PythonLangchainLangChain 概览
LangChain 概览
此页内容

文档语言
Python
TypeScript
深入探索
LangChain 中文教程
LangChain 快速入门
LangChain 概览
智能代理
多智能体系统
探索开发工具
体验AI智能工具
学习编程课程
智能体 = 模型 + 运行框架。 LangChain 提供 create_agent：一个简洁且高度可配置的运行框架。运行框架是围绕模型循环的一切组件，包括 Prompt、工具以及任何用于塑造行为的中间件。你可以从这些基础单元开始，精确组合用例所需的能力。支持 OpenAI、Anthropic、Google 等提供商。 智能代理

提示

LangChain、LangGraph 与 Deep Agents 的区别

如果需要一个“开箱即用”的智能体，请从 Deep Agents 开始。它具备自动上下文压缩、虚拟文件系统和子智能体创建等功能。Deep Agents 构建在 LangChain 智能体之上，你也可以直接使用后者。

如果需要高度可定制、能够轻松适配具体用例和数据的运行框架，请使用 LangChain（create_agent）。

如果有组合确定性工作流与智能体工作流的高级需求，请使用底层编排框架 LangGraph。

使用 LangSmith 追踪、调试和评估由这些框架构建的智能体。请按照追踪快速入门完成设置。我们还建议设置 LangSmith Engine，它会监控追踪记录、检测问题并提出修复建议。


创建智能体
深入探索
Python 更新日志
探索工程技术
Python 设计理念
以下示例演示如何使用自定义工具创建一个简单的 LangChain 智能体： LangChain中文教程

OpenAIGoogle GeminiClaude (Anthropic)OpenRouterFireworksBasetenOllamaAzureAWS BedrockHuggingFace

# pip install -qU langchain "langchain[openai]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city} 总是阳光明媚！"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="你是一名乐于助人的助手",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样？"}]}
)
print(result["messages"][-1].content_blocks)
请参阅安装说明和快速入门指南，开始使用 LangChain 构建自己的智能体和应用。 LangChain中文教程

提示

深入探索
订阅生产力软件
LangGraph 教程
学习脚本语言
使用 LangSmith 追踪请求、调试智能体行为并评估输出。设置 LANGSMITH_TRACING=true 和 API 密钥即可开始使用。 智能代理


核心优势
标准模型接口
    使用统一接口调用不同提供商的聊天模型、嵌入模型等。只需极少的代码改动即可切换模型，并在需求演进时保持应用的可移植性。
高度可配置的运行框架
    从最简运行框架 `create_agent` 开始，通过中间件逐步添加能力。只组合用例所需的功能，包括护栏、重试、路由和自定义工具策略。
构建在 LangGraph 之上
    LangChain 智能体构建在 LangGraph 之上，因此可以利用 LangGraph 的持久执行、人在回路、持久化等能力。
使用 LangSmith 调试
    在一个位置检查追踪记录、工具调用、状态转换和延迟。利用执行数据发现故障模式、评估质量并改进智能体行为。
深入探索
核心组件
软件框架
运行框架
最近更新2026/10/10 20:50
上一页
LangChain v1.0
下一页
开始使用

LangChain 中文文档 | © 2025–至今 LangChain-doc.cn 中文社区
关于我们 隐私政策 服务条款