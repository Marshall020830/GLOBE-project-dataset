import os
from openai import OpenAI
import pandas as pd

trace = ["Assertiveness", "Future_orientation", "Gender_egalitarianism", "Humane_orientation",
         "In-group_collectivism", "Institutional_collectivism", "Performance_orientation",  "Power_distance", "Uncertainty_avoidance"]

for root in trace:
    input_file = root + "/" + root + "_random.csv"
    output_file = root + "/" + root + "_random_DSV3.csv"
    # 初始化 DeepSeek 客户端
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    # 尝试加载已有输出文件（支持断点续跑）
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print("已加载已有结果文件，启用断点续跑模式")
    else:
        df = pd.read_csv(input_file)
        df['Response'] = ""
        print("加载原始输入文件")

    # 找出尚未回答的索引
    pending_indices = df[(df['Response'].isnull()) | (df['Response'].str.strip() == "") | (df['Response'] == "ERROR")].index.tolist()
    print(f"待处理问题数：{len(pending_indices)}")

    # 遍历待处理问题
    for count, i in enumerate(pending_indices, start=1):
        question = df.at[i, 'Translated Question']
        print(f"\n[{i}] Question: {question}")

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "I'm going to ask you a series of questions. Please just make a simple choice—no extra explanation is needed. If you chose the previous answer, please output A, otherwise output B. Answer in the same language as the question."},
                    {"role": "user", "content": question},
                ],
                stream=False
            )
            answer = response.choices[0].message.content.strip()
            print(f"Response: {answer}")
        except Exception as e:
            answer = "ERROR"
            print(f"Error at index {i}: {e}")

        # 写入当前结果
        df.at[i, 'Response'] = answer

        # 每25条保存一次
        if count % 25 == 0 or count == len(pending_indices):
            df.to_csv(output_file, index=False)
            print(f"进度已保存（共处理 {count} 条，当前 index = {i}）")

    print("\n🎉 所有问题处理完成，结果已保存。")
    df.to_csv(output_file, index=False)
print("\n任务结束。")
