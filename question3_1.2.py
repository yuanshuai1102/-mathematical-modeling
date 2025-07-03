import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('train.csv',encoding='GBK')

# 处理缺失值
data = data.dropna()

# 提取特征和标签
features = data.drop(columns=['id', '洪水概率'])
labels = data['洪水概率']

# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 调整数据形状以适应LSTM
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# 定义 LSTM 模型
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(10))  # 输出特征为 10 维
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, validation_data=(X_test_lstm, y_test), verbose=2, shuffle=False)

# 使用LSTM提取特征
X_train_lstm_features = lstm_model.predict(X_train_lstm)
X_test_lstm_features = lstm_model.predict(X_test_lstm)


# 使用XGBoost进行预测
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
xgb_model.fit(X_train_lstm_features, y_train)

# 预测结果
y_pred_proba = xgb_model.predict(X_test_lstm_features)
y_pred_class = (y_pred_proba > 0.5).astype(int)

# 将 y_test 转换为二元分类值
y_test_class = (y_test > 0.5).astype(int)

# 计算评估指标
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
mse = mean_squared_error(y_test, y_pred_proba)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_proba)
mape = mean_absolute_percentage_error(y_test, y_pred_proba)

# 打印评估指标
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')

# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_test_class, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# 绘制 Precision-Recall 曲线
precision_values, recall_values, _ = precision_recall_curve(y_test_class, y_pred_proba)
plt.figure()
plt.plot(recall_values, precision_values, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# 绘制实际值和预测值的对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='实际')
plt.plot(y_pred_proba, label='预测')
plt.legend()
plt.title('lstm-xgboost预测效果图')
plt.xlabel('Sample Index')
plt.ylabel('Flood Probability')
plt.show()
