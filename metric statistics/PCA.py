import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})

# 加载数据
data = pd.read_csv('company_data_v2.csv')

# 删除与目标变量无关的列
data_cleaned = data.drop(['id', 'year'], axis=1)

# 特征和目标变量
features = data_cleaned.drop(columns=['Financialfraud'])
target = data_cleaned['Financialfraud']

# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA
pca = PCA()
pca.fit(features_scaled)

# 获取每个特征的贡献度（权重）
explained_variance_ratio = pca.explained_variance_ratio_

# 保存每个特征的贡献度
feature_names = features.columns
pca_contribution = pd.DataFrame({
    'Feature': feature_names,
    'Contribution': explained_variance_ratio
})

# 将特征分为两组
num_features = len(feature_names)
mid_point = num_features // 2
pca_contribution_group1 = pca_contribution.iloc[:mid_point]
pca_contribution_group2 = pca_contribution.iloc[mid_point:]

def plot_group_bar_charts(group_df, group_name, ax):
    """ 绘制给定组的条形图 """
    sns.barplot(x='Contribution', y='Feature', data=group_df, palette='viridis', ax=ax)
    ax.set_title(f'PCA Feature Contribution (Bar Plot) - {group_name}', fontsize=16)
    ax.set_xlabel('Contribution', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

# 绘制两个条形图并将它们拼在一起
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
plot_group_bar_charts(pca_contribution_group1, 'Group 1', axs[0])
plot_group_bar_charts(pca_contribution_group2, 'Group 2', axs[1])

plt.tight_layout()
plt.savefig('pca_feature_contributions_combined.png', dpi=300)
plt.show()