import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from mealpy.swarm_based.GWO import OriginalGWO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

# 读取数据
data = pd.read_csv('train.csv', encoding='GBK')

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

# 定义LSTM模型
def create_lstm_model(lstm_units, dropout_rate):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10))  # 输出特征维度
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 训练LSTM模型并提取特征
def extract_features(lstm_units, dropout_rate):
    lstm_model = create_lstm_model(lstm_units, dropout_rate)
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, verbose=0, shuffle=False)
    X_train_features = lstm_model.predict(X_train_lstm)
    X_test_features = lstm_model.predict(X_test_lstm)
    return X_train_features, X_test_features

# 定义XGBoost模型并评估
def evaluate_model(params):
    lstm_units = int(params[0])
    dropout_rate = params[1]
    max_depth = int(params[2])
    learning_rate = params[3]
    n_estimators = int(params[4])

    X_train_features, X_test_features = extract_features(lstm_units, dropout_rate)

    xgb_model = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective='reg:squarederror')
    xgb_model.fit(X_train_features, y_train)
    y_pred_proba = xgb_model.predict(X_test_features)
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score((y_test > 0.5).astype(int), y_pred_class)
    return -accuracy

# 定义优化参数范围
problem = {
    "fit_func": evaluate_model,
    "lb": [10, 0.1, 3, 0.01, 50],   # Lower bounds for lstm_units, dropout_rate, max_depth, learning_rate, n_estimators
    "ub": [100, 0.5, 10, 0.3, 200], # Upper bounds for lstm_units, dropout_rate, max_depth, learning_rate, n_estimators
    "minmax": "min",
}

# 初始化灰狼优化算法
gwo_model = OriginalGWO(problem, epoch=10, pop_size=20)

# 运行优化
best_position, best_fitness = gwo_model.solve()

# 打印最佳参数
print(f"Best Parameters: {best_position}, Best Fitness: {best_fitness}")

# 使用最佳参数训练最终模型
best_lstm_units = int(best_position[0])
best_dropout_rate = best_position[1]
best_max_depth = int(best_position[2])
best_learning_rate = best_position[3]
best_n_estimators = int(best_position[4])

X_train_features, X_test_features = extract_features(best_lstm_units, best_dropout_rate)

final_xgb_model = xgb.XGBRegressor(max_depth=best_max_depth, learning_rate=best_learning_rate, n_estimators=best_n_estimators, objective='reg:squarederror')
final_xgb_model.fit(X_train_features, y_train)

# 预测结果
y_pred_proba = final_xgb_model.predict(X_test_features)
y_pred_class = (y_pred_proba > 0.5).astype(int)

# 计算评估指标
precision = precision_score((y_test > 0.5).astype(int), y_pred_class)
recall = recall_score((y_test > 0.5).astype(int), y_pred_class)
f1 = f1_score((y_test > 0.5).astype(int), y_pred_class)
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
fpr, tpr, _ = roc_curve((y_test > 0.5).astype(int), y_pred_proba)
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
precision_values, recall_values, _ = precision_recall_curve((y_test > 0.5).astype(int), y_pred_proba)
plt.figure()
plt.plot(recall_values, precision_values, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

