import os
import requests
import pandas as pd

trace = ["Assertiveness", "Future_orientation", "Gender_egalitarianism", "Humane_orientation",
         "In-group_collectivism", "Institutional_collectivism", "Performance_orientation",  "Power_distance", "Uncertainty_avoidance"]

for root in trace:
    input_file = root + "/" + root + "_random_value.csv"
    output_file = root + "/" + root + "_random_value_GPT35.csv"

    # 尝试加载已有的结果文件
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"已加载已有结果文件{output_file}，启用断点续跑模式")
    else:
        df = pd.read_csv(input_file)
        df['Response'] = ""  # 添加空的结果列
        print(f"加载原始输入文件{input_file}")

    # 获取要处理的行索引：为空或是 ERROR 的才处理
    pending_indices = df[(df['Response'].isnull()) | (df['Response'].str.strip() == "") | (df['Response'] == "ERROR")].index.tolist()
    print(f"当前维度需要处理的问题数：{len(pending_indices)}")

    # 开始请求
    for count, i in enumerate(pending_indices, start=1):
        question = df.at[i, 'Translated Question']
        print(f"\n[{i}] Question: {question}")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "I'm going to ask you a series of questions. Please just make a simple choice—no extra explanation is needed. If you chose the previous answer, please output A, otherwise output B. Just answer A or B."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            print(f"Response: {answer}")
        except Exception as e:
            answer = "ERROR"
            print(f"Error at index {i}: {e}")

        # 写入回答
        df.at[i, 'Response'] = answer

        # 每25条保存一次
        if count % 25 == 0 or count == len(pending_indices):
            df.to_csv(output_file, index=False)
            print(f"已保存至 {output_file}（处理到第 {i} 行）")
    df.to_csv(output_file, index=False)
    print("\n所有待处理问题已完成并保存。")
print("\n任务结束。")
