import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.utils import resample

# 读取数据
df = pd.read_csv('All.csv')

# 假设数据列包括 'Model', 'Precision', 'Recall', 'F1-Score', 'Accuracy'
models = df['Model'].unique()
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

# Kruskal-Wallis H 检验
def kruskal_test(metric_values):
    grouped_values = [metric_values[df['Model'] == model].values for model in models]
    stat, p_value = kruskal(*grouped_values)
    return stat, p_value

# Bootstrap 方法
def bootstrap_test(metric_values, n_iterations=1000, ci=95):
    boot_means = []
    for _ in range(n_iterations):
        sample = resample(metric_values)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

# 逐指标进行检验
kruskal_results = []
bootstrap_results = []

for metric in metrics:
    metric_values = [df[df['Model'] == model][metric].values for model in models]

    # Kruskal-Wallis H 检验
    stat, p_value = kruskal_test(df[metric])
    kruskal_results.append({
        'Metric': metric,
        'Kruskal-Wallis H Statistic': stat,
        'Kruskal-Wallis H p-value': p_value
    })

    # Bootstrap 方法
    for model in models:
        lower, upper = bootstrap_test(df[df['Model'] == model][metric].values)
        bootstrap_results.append({
            'Metric': metric,
            'Metric Type': model,
            'CI Lower': lower,
            'CI Upper': upper
        })

# 将结果保存为 CSV 文件
kruskal_results_df = pd.DataFrame(kruskal_results)
bootstrap_results_df = pd.DataFrame(bootstrap_results)

kruskal_results_df.to_csv('kruskal_results.csv', index=False)
bootstrap_results_df.to_csv('bootstrap_results.csv', index=False)

print("Kruskal-Wallis H 检验结果已保存到 kruskal_results.csv 文件中。")
print("Bootstrap 方法结果已保存到 bootstrap_results.csv 文件中。")