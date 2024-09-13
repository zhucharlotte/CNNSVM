import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# 加载数据
data = pd.read_csv('company_data_v2.csv')

# 删除与目标变量无关的列
data_cleaned = data.drop(['id', 'year'], axis=1)

# 进行过采样来平衡数据集
data_majority = data_cleaned[data_cleaned.Financialfraud == 0]
data_minority = data_cleaned[data_cleaned.Financialfraud == 1]
data_minority_upsampled = resample(data_minority, replace=True, n_samples=len(data_majority), random_state=42)
data_balanced = pd.concat([data_majority, data_minority_upsampled])

# 分割数据
X = data_balanced.drop('Financialfraud', axis=1)
y = data_balanced['Financialfraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 调整数据维度以适应CNN模型
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# 构建 CNN 模型
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 添加早停法
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 训练模型
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 使用 SVM 对 CNN 模型的输出进行分类
svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
svm_model.fit(model.predict(X_train_reshaped), y_train)
y_pred_svm = svm_model.predict(model.predict(X_test_reshaped))

# 打印分类报告
print(classification_report(y_test, y_pred_svm))

# 计算混淆矩阵并可视化
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap='Blues', cbar=False)
plt.title('Confusion Matrix - CNN-SVM')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 计算ROC曲线
y_proba_svm = svm_model.predict_proba(model.predict(X_test_reshaped))[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_svm)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - CNN-SVM')
plt.legend(loc="lower right")
plt.show()
