import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('All.csv')

# 计算描述性统计
def describe_data(df):
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    rows = []

    for model, group in df.groupby('Model'):
        for metric in metrics:
            metric_data = group[metric]
            stats_data = {
                'Model': model,
                'Metric': metric,
                'Count': len(metric_data),
                'Mean': metric_data.mean(),
                'Std': metric_data.std(),
                'Median': metric_data.median(),
                'Max': metric_data.max(),
                'Min': metric_data.min()
            }
            rows.append(stats_data)

    stats_df = pd.DataFrame(rows)
    return stats_df

# 打印描述性统计
stats_df = describe_data(df)
print(stats_df)

# 将描述性统计保存到CSV文件
stats_df.to_csv('descriptive_statistics.csv', index=False)

# 绘制箱线图
def plot_boxplots(df):
    plt.figure(figsize=(14, 10))

    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='Model', y=metric, data=df)
        plt.title(f'Boxplot of {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric)

    plt.tight_layout()
    plt.show()

# 绘制箱线图
plot_boxplots(df)