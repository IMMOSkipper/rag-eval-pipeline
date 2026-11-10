import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import warnings
warnings.filterwarnings(
    "ignore",
    # message=r"Importing .+ from 'ragas\.metrics' is deprecated",
    category=DeprecationWarning,
)

import time
import sys
import json
import numpy as np
from dotenv import load_dotenv
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from app.generate import ask_rag
from ragas.llms import LangchainLLMWrapper

# 初始化裁判模型（设置 seed 保证相对稳定）
judge_llm = ChatOpenAI(
    model_name=os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)
judge = LangchainLLMWrapper(judge_llm, bypass_n=True)

judge_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"}
)

def run_eval(sample_times=3):
    print(f"🚀 开始读取测试集并生成RAG回答...")
    valid_questions, valid_answers, valid_contexts, valid_references = [], [], [], []
    refuse_total, refuse_correct = 0, 0

    with open("eval/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            q = data["query"]
            expected = data["expected_behavior"]

            start_t = time.time()
            ans, ctx = ask_rag(q)
            is_refuse = "未检索到有效内容" in ctx or "无法回答" in ans or "抱歉" in ans
            latency = time.time() - start_t

            if expected == "refuse":
                refuse_total += 1
                if is_refuse:
                    refuse_correct += 1
                    print(f"✅️ [拒答成功] {q} | 耗时: {latency:.2f}s")
                else:
                    print(f"❌️ [拒答失败] {q} | 耗时: {latency:.2f}s")
            elif expected == "answer":
                if not is_refuse:
                    print(f"✅️ [正常回答] {q} | 耗时: {latency:.2f}s")
                    valid_questions.append(q)
                    valid_answers.append(ans)
                    valid_contexts.append([ctx])
                    valid_references.append(data.get("reference", ""))
                else:
                    print(f"❌️ [漏答] {q} | 耗时: {latency:.2f}s")
            
            # 接口异常判断：如果大模型调用失败，记录为空，不进入Ragas评测
            if "调用大模型失败" in ans:
                print(f"⚠️ 跳过 {q} (API报错)")
                continue       

    # if not valid_questions:
    #     print("❌️ 测试集中没有有效数据，请检查API状态。")
    #     return

    print("\n" + "="*40)
    print("🛡️ 拒答准确率（Refusal Accuracy）")
    print("="*40)
    if refuse_total > 0:
        print(f"🎯 成功拦截: {refuse_correct}/{refuse_total}, 准确率: {(refuse_correct/refuse_total)*100:.1f}%")

    if valid_questions:
        print(f"\n📊 开始对有效回答进行 Ragas 质量打分 (外层多重采样 {sample_times} 次)...")
        # 构建Ragas数据集
        data_dict = {
            "user_input": valid_questions,
            "response": valid_answers,
            "retrieved_contexts": valid_contexts,
            "reference": valid_references
        }
        dataset = Dataset.from_dict(data_dict)
        scores_dict = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": []}
        for i in range(sample_times):
            print(f" 🔄 第 {i+1}/{sample_times} 次采样评测中...")
            try:
                result = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=judge,
                    embeddings=judge_embeddings
                )
                df = result.to_pandas()
                missing = [
                    metric for metric in scores_dict
                    if metric not in df.columns or df[metric].isna().any()
                ]
                if missing:
                    print(f"  ⚠️ 本次采样有指标缺失，不计入均值: {missing}")
                    continue
                for metric in scores_dict:
                    scores_dict[metric].append(float(df[metric].mean()))
            except Exception as e:
                print(f"    ⚠️ 本次采样部分指标失败: {e}")

        print("\n" + "="*40)
        print("🏆 Ragas 最终评测报告 (多重采样平均)")
        print("="*40)

        report_data = {}
        for metric, scores in scores_dict.items():
            if len(scores) > 0:
                mean_val = round(float(np.mean(scores)), 4)
                std_val = round(float(np.std(scores)), 4)
                print(f"📈 {metric: <20}: 均值 {mean_val:.4f} (标准差 ±{std_val:.4f}) | 原始数据: {[round(float(s),2) for s in scores]}")
                report_data[metric] = {"mean": round(mean_val, 4), "std": round(std_val, 4), "runs": scores}
            else:
                print(f"❌️ {metric}: 未能计算(可能由于接口返回 400)")

        report_data["refusal_accuracy"] = round(refuse_correct / refuse_total, 4) if refuse_total > 0 else 0.0
        with open("eval/ragas_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print("\n✅️ 评测结果已保存至 eval/ragas_report.json")

if __name__ == "__main__":
    run_eval(sample_times=3)