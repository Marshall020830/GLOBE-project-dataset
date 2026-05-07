import os
from openai import OpenAI
import pandas as pd
from multiprocessing import Process

trace = [
    "Assertiveness", "Future_orientation", "Gender_egalitarianism", "Humane_orientation",
    "In-group_collectivism", "Institutional_collectivism", "Performance_orientation",
    "Power_distance", "Uncertainty_avoidance"
]

def process_dimension(root):
    input_file = root + "/" + root + "_random.csv"
    output_file = root + "/" + root + "_random_DSR1.csv"

    # 初始化 DeepSeek 客户端
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    # 尝试加载已有输出文件（支持断点续跑）
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"[{root}] 已加载已有结果文件，启用断点续跑模式")
    else:
        df = pd.read_csv(input_file)
        df['Response'] = ""
        print(f"[{root}] 加载原始输入文件")

    # 找出尚未回答的索引
    pending_indices = df[(df['Response'].isnull()) | (df['Response'].str.strip() == "") | (df['Response'] == "ERROR")].index.tolist()
    print(f"[{root}] 待处理问题数：{len(pending_indices)}")

    for count, i in enumerate(pending_indices, start=1):
        question = df.at[i, 'Translated Question']
        print(f"[{root}] [{i}] Question: {question}")

        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "I'm going to ask you a series of questions. Please just make a simple choice—no extra explanation is needed. If you chose the previous answer, please output A, otherwise output B. Answer in the same language as the question."},
                    {"role": "user", "content": question},
                ],
                stream=False
            )
            answer = response.choices[0].message.content.strip()
            print(f"[{root}] Response: {answer}")
        except Exception as e:
            answer = "ERROR"
            print(f"[{root}] Error at index {i}: {e}")

        df.at[i, 'Response'] = answer

        if count % 25 == 0 or count == len(pending_indices):
            df.to_csv(output_file, index=False)
            print(f"[{root}] 进度已保存（共处理 {count} 条，当前 index = {i}）")

    df.to_csv(output_file, index=False)
    print(f"[{root}] 🎉 所有问题处理完成，结果已保存。")


if __name__ == "__main__":
    processes = []

    for dimension in trace:
        p = Process(target=process_dimension, args=(dimension,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ 所有维度处理完成，任务结束。")
