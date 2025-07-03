import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
train_data = pd.read_csv('train.csv',encoding='GBK')
test_data = pd.read_csv('test.csv',encoding='GBK')

# 提取特征和标签
X = train_data.drop(columns=['id', '洪水概率'])
y = train_data['洪水概率']

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data.drop(columns=['id','洪水概率']))

# LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# 将数据调整为LSTM所需的形状
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 训练LSTM模型
model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, validation_data=(X_val_lstm, y_val))

# XGBoost模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train_scaled, y_train)

# 预测
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_lstm = model.predict(X_test_lstm).flatten()

# 融合两种模型的预测结果
final_predictions = (y_pred_xgb + y_pred_lstm) / 2

# 将预测结果生成新的CSV文件
output = pd.DataFrame({'id': test_data['id'], '洪水概率': final_predictions})
output.to_csv('D:/pycharm/数学建模/mathematics modeling/flood_predictions.csv', index=False,encoding='utf-8')

# 绘制直方图和折线图
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.hist(output['洪水概率'], bins=50, color='pink', alpha=0.5)
plt.title('预测洪水概率的直方图')
plt.xlabel('洪水概率')
plt.ylabel('频率')

plt.subplot(1, 2, 2)
plt.plot(output['洪水概率'], color='#ADD8E6')
plt.title('预测洪水概率折线图')
plt.xlabel('样本序号')
plt.ylabel('洪水概率')

plt.show()

# 分布是否服从正态分布
stat, p = normaltest(output['洪水概率'])
print(f'Statistics={stat}, p={p}')
if p > 0.05:
    print('样本数据服从正态分布')
else:
    print('样本数据不服从正态分布')

