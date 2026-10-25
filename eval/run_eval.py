import json
import os
import sys
import time

# 将项目根目录加入路径，方便导入 app 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.generate import ask_rag

EVAL_FILE = "eval/questions.jsonl"

def run_evaluation(model_name="deepseek-v4.1-flash-1017"):
    print(f"🧪 开始运行 RAG 自动化评测（模型：{model_name}）")
    print("=" * 50)

    total = 0
    pass_count = 0

    # 读取评测集
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query = data["query"]
            excepted = data["excepted_behavior"]

            total += 1
            print(f"\n[{total}] 问题：{query}")
            print(f"    预期行为：{excepted}")

            # 调用RAG（为了省时间，这里调用写好的函数）
            # 可以给函数加点小改造，让它能返回是否命中上下文
            # 但为了简单，通过判断回答里是否有“无法回答”来粗略定性
            answer = ask_rag(query, model_name=model_name)

            is_refuse = any(keyword in answer for keyword in ["无法回答", "没有找到", "没有提供"])

            if excepted == "answer" and not is_refuse:
                print(f"    ✅️ 结果：通过（成功回答）")
                pass_count += 1
            elif excepted == "refuse" and is_refuse:
                print(f"    ✅️ 结果：通过（成功拒答）")
                pass_count += 1
            else:
                print(f"    ❌️ 结果：失败（不符合预期。模型实际输出：{answer[:50]}...）")

            time.sleep(1) # 稍微停顿，防止触发限流

    print("\n" + "=" * 50)
    print(f"🏆 评测完成！总题数：{total}, 通过：{pass_count}, 准确率：{pass_count/total * 100:.1f}%")

if __name__ == "__main__":
    run_evaluation(model_name="deepseek-v4.1-flash-1017")