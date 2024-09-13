import pandas as pd
import matplotlib.pyplot as plt

# CSV 文件名和标题
files = [
    ('Non-Fraud Business Operating.csv', 'Non-Fraud Business Operating'),
    ('Non-Fraud Corporate Governance.csv', 'Non-Fraud Corporate Governance'),
    ('Non-Fraud Financial Indicators.csv', 'Non-Fraud Financial Indicators')
]

# 创建一个 1x3 的图形网格
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (file, title) in zip(axes, files):
    df = pd.read_csv(file, index_col=0).T
    variables = df.index
    box_data = []

    for _, row in df.iterrows():
        box_data.append([
            row['minimum value'],
            row['p25'],
            row['median value'],
            row['p75'],
            row['maximum value']
        ])

    # 绘制箱线图
    ax.boxplot(box_data, labels=variables, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               medianprops=dict(color='red'))

    # 设置标题和标签
    ax.set_title(f'Box Plot - {title}')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Values')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

# 保存图像
plt.savefig('combined_boxplots.png')

plt.show()