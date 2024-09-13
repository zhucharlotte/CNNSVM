import pandas as pd
import matplotlib.pyplot as plt

# CSV 文件名
file = 'Fraud Accounting Supervision.csv'

# 读取 CSV 文件
df = pd.read_csv(file)

# 只选择 'Big4' 和 'Opinion' 两列
columns_to_plot = ['Big4', 'Opinion']
df = df[columns_to_plot]

# 去掉文件后缀以便用作图标题
title = file.split('.')[0]

# 创建一个 1x2 的图形网格
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, column in enumerate(df.columns):
    # 计算每个变量的值
    counts = df[column].value_counts()
    labels = [f'{val}: {count}' for val, count in counts.items()]

    # 颜色列表与标签
    colors = ['lightblue' if val == 1 else 'lightgreen' for val in counts.index]
    legend_labels = [f'{val}' for val in counts.index]

    # 绘制饼状图
    wedges, texts, autotexts = axes[i].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)

    # 添加图例
    axes[i].legend(wedges, legend_labels, title="Values", loc="best", bbox_to_anchor=(1, 1))

    # 设置图表标题
    axes[i].set_title(column)

plt.suptitle(title)
plt.tight_layout()

# 保存图像
plt.savefig(f'{title}.png')

plt.show()