import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import sys
import numpy as np
from dotenv import load_dotenv
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.generate import ask_rag

# 初始化裁判模型（设置 seed 保证相对稳定）
judge_llm = ChatOpenAI(
    model_name=os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    model_kwargs={"seed": 42}
)

judge_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"}
)

def run_eval(sample_times=3):
    print(f"🚀 开始读取测试集并生成RAG回答...")
    questions, answers, contexts, references = [], [], [], []

    with open("eval/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            q = data["query"]
            ref = data.get("reference", "无法回答")

            start_t = time.time()
            rag_result = ask_rag(q)
            if isinstance(rag_result, tuple) and len(rag_result) == 2:
                ans, ctx = rag_result
            else:
                ans = rag_result
                ctx = "未检索到有效内容"
            latency = time.time() - start_t

            # 接口异常判断：如果大模型调用失败，记录为空，不进入Ragas评测
            if "调用大模型失败" in ans:
                print(f"⚠️ 跳过 {q} (API报错)")
                continue
                
            print(f"✅️ {q} | 耗时: {latency:.2f}s")
            questions.append(q)
            answers.append(ans)
            contexts.append([ctx])
            references.append(ref)

    if not questions:
        print("❌️ 测试集中没有有效数据，请检查API状态。")
        return

    valid_questions, valid_answers, valid_contexts, valid_references = [], [], [], []
    for q, a, c, r in zip(questions, answers, contexts, references):
        if ["未检索到有效内容"] not in c:
            valid_questions.append(q)
            valid_answers.append(a)
            valid_contexts.append(c)
            valid_references.append(r)

    # 构建Ragas数据集
    data_dict = {
        "user_input": valid_questions,
        "response": valid_answers,
        "retrieved_contexts": valid_contexts,
        "reference": valid_references
    }
    dataset = Dataset.from_dict(data_dict)

    print(f"\n📊 正在请求 Ragas 裁判模型打分（多重采样 {sample_times} 次）...")

    # 存储每次采样的结果
    scores_dict = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}

    for i in range(sample_times):
        print(f" 🔄 第 {i+1}/{sample_times} 次采样评测中...")
        try:
            # 运行评测（使用大模型作为裁判）
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=judge_llm,
                embeddings=judge_embeddings
            )
            result_dict = result.to_pandas().to_dict(orient="records")[0]
            # 兼容 Ragas 不同版本的取值方式
            for metric in scores_dict.keys():
                val = result_dict.get(metric, 0.0)
                if np.isnan(val): 
                    val = 0.0
                scores_dict[metric].append(val)
        except Exception as e:
            print(f"    ⚠️ 本次采样部分指标失败: {e}")

    # 汇总计算结果
    final_report = {}
    print("\n" + "="*40)
    print("🏆 Ragas最终评测报告")
    print("="*40)

    for metric, scores in scores_dict.items():
        if len(scores) > 0:
            mean_val = round(float(np.mean(scores)), 4)
            std_val = round(float(np.std(scores)), 4)
            print(f"📈 {metric: <20}: 均值 {mean_val:.4f} (方差 ±{std_val:.4f}) | 原始数据: {[round(s,2) for s in scores]}")
            final_report[metric] = {"mean": mean_val, "std": std_val, "runs": scores}
        else:
            print(f"❌️ {metric}: 未能计算(可能由于接口返回 400)")


    with open("eval/ragas_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print("✅️ 评测结果已保存至 eval/ragas_report.json")

if __name__ == "__main__":
    run_eval(sample_times=3)