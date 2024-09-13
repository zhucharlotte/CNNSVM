import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 读取 CSV
df = pd.read_csv('All.csv')

# 进行方差分析 (ANOVA)
def perform_anova(df, metric):
    models = df['Model'].unique()
    data = [df[df['Model'] == model][metric] for model in models]
    f_stat, p_value = stats.f_oneway(*data)
    return f_stat, p_value

# 正态性检验
def check_normality(df, metric):
    models = df['Model'].unique()
    p_values = {}
    for model in models:
        group_data = df[df['Model'] == model][metric]
        stat, p_value = stats.shapiro(group_data)
        p_values[model] = p_value
    return p_values

# 方差齐性检验
def check_homogeneity_of_variances(df, metric):
    models = df['Model'].unique()
    group_data = [df[df['Model'] == model][metric] for model in models]
    stat, p_value = stats.levene(*group_data)
    return stat, p_value

# 绘制箱线图
def plot_boxplot(df, metric):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y=metric, data=df)
    plt.title(f'Boxplot of {metric}')
    plt.show()

# 保存结果为 CSV
results = []

# 指标列表
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

# 对每个指标进行方差分析、正态性检验和方差齐性检验，并可视化
for metric in metrics:
    # 方差分析
    f_stat, p_value = perform_anova(df, metric)
    results.append([metric, 'ANOVA', f_stat, f"{p_value:.6f}"])

    # 正态性检验
    normality_p_values = check_normality(df, metric)
    for model, p_value in normality_p_values.items():
        results.append([metric, f'{model} Normality', '-', f"{p_value:.6f}"])

    # 方差齐性检验
    levene_stat, levene_p_value = check_homogeneity_of_variances(df, metric)
    results.append([metric, "Levene's Test", levene_stat, f"{levene_p_value:.6f}"])

    # 绘制箱线图
    plot_boxplot(df, metric)

# 保存结果到 CSV 文件
results_df = pd.DataFrame(results, columns=['Metric', 'Test', 'Statistic', 'p-value'])
results_df.to_csv('analysis_results.csv', index=False)