import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# 加载数据
data = pd.read_csv('company_data_v2.csv')

# 删除与目标变量无关的列
df = data.drop(['id', 'year'], axis=1)

# 特征和目标变量
features = df.drop(columns=['Financialfraud'])
target = df['Financialfraud']

# 绘制相关性热图
correlation_matrix = features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Feature Correlation Heatmap')

# 输出目录
output_dir = 'output'  # 确保你设置了正确的输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.show()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=500)  # 增加dpi参数以提高图像清晰度
plt.close()
