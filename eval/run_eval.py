import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import time
import sys
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.generate import ask_rag

load_dotenv()

judge_llm = ChatOpenAI(
    model_name=os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

judge_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"}
)

def run_eval():
    questions = []
    answers = []
    contexts = []

    with open("eval/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["expected_behavior"] == "answer":
                questions.append(data["query"])
                ans, ctx = ask_rag(data["query"])
                answers.append(ans)
                contexts.append([ctx])

    # 构建Ragas数据集
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts
    }
    dataset = Dataset.from_dict(data)

    print("📊 正在请求 Ragas 裁判模型打分，请稍候...")

    # 运行评测（使用大模型作为裁判）
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness], #, answer_relevancy
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    print("🎉 评测完成！")
    print(result)

    try:
        result_dict = result.to_pandas().to_dict(orient="records")
        with open("eval/ragas_report.json", "w") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print("✅️ 评测结果已保存至 eval/ragas_report.json")
    except Exception as e:
        print(f"⚠️ 保存 JSON 失败，但不影响评测结果: {e}")
    

if __name__ == "__main__":
    run_eval()