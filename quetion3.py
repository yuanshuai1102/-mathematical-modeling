import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
train_data = pd.read_csv('train.csv',encoding='GBK')

# 检查数据基本信息
print(train_data.info())
print(train_data.describe())

# 处理缺失值（此处简单处理为填充均值，可根据具体情况调整）
train_data.fillna(train_data.mean(), inplace=True)

# 分离特征和目标变量
X = train_data.drop(columns=['id', '洪水概率'])  # 假设 'flood_probability' 是洪水发生的概率
y = train_data['洪水概率']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# 转换为分类任务，选择合适的阈值
threshold = 0.5
y_pred_class = (y_pred >= threshold).astype(int)
y_test_class = (y_test >= threshold).astype(int)

# 分类评估指标
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)

# ROC曲线和AUC
fpr, tpr, _ = roc_curve(y_test_class, y_pred)
roc_auc = auc(fpr, tpr)

# 打印评估指标
print(f'Model Mean Squared Error (MSE): {mse}')
print(f'Model Root Mean Squared Error (RMSE): {rmse}')
print(f'Model Mean Absolute Error (MAE): {mae}')
print(f'Model Mean Absolute Percentage Error (MAPE): {mape}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
