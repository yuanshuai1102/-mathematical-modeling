import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

# 加载数据
train_df = pd.read_csv('train.csv',encoding='GBK')

# 查看数据结构
print(train_df.head())

# 检查缺失值
print(train_df.isnull().sum())

# 简单的数据预处理
train_df.fillna(train_df.mean(), inplace=True)

# 假设我们从问题1的分析中得到了以下重要特征
important_features = ['地形排水', '人口得分', '海岸脆弱性', '政策因素', '无效防灾']

# 特征和目标变量
X = train_df[important_features]
y = train_df['洪水概率']

# 将目标变量转为二分类
y_binary = (y >= 0.5).astype(int)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数，使用CPU
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'eta': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'silent': 1,
    'tree_method': 'hist',  # 使用CPU
    'predictor': 'cpu_predictor'
}

# 训练模型
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# 预测
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob >= 0.5).astype(int)

# 评估模型
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred_prob)
rmse = mean_squared_error(y_test, y_pred_prob, squared=False)
mae = mean_absolute_error(y_test, y_pred_prob)
mape = np.mean(np.abs((y_test - y_pred_prob) / y_test)) * 100
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'ROC AUC: {roc_auc}')

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 调整使用5个关键指标后的模型
# 重新训练和评估
key_features = important_features[:5]
X_key = train_df[key_features]

X_train_key, X_test_key, y_train_key, y_test_key = train_test_split(X_key, y_binary, test_size=0.2, random_state=42)

dtrain_key = xgb.DMatrix(X_train_key, label=y_train_key)
dtest_key = xgb.DMatrix(X_test_key, label=y_test_key)

bst_key = xgb.train(params, dtrain_key, num_round)

y_pred_prob_key = bst_key.predict(dtest_key)
y_pred_key = (y_pred_prob_key >= 0.5).astype(int)

precision_key = precision_score(y_test_key, y_pred_key)
recall_key = recall_score(y_test_key, y_pred_key)
f1_key = f1_score(y_test_key, y_pred_key)
mse_key = mean_squared_error(y_test_key, y_pred_prob_key)
rmse_key = mean_squared_error(y_test_key, y_pred_prob_key, squared=False)
mae_key = mean_absolute_error(y_test_key, y_pred_prob_key)
mape_key = np.mean(np.abs((y_test_key - y_pred_prob_key) / y_test_key)) * 100
roc_auc_key = roc_auc_score(y_test_key, y_pred_prob_key)

print(f'Precision with key features: {precision_key}')
print(f'Recall with key features: {recall_key}')
print(f'F1-score with key features: {f1_key}')
print(f'MSE with key features: {mse_key}')
print(f'RMSE with key features: {rmse_key}')
print(f'MAE with key features: {mae_key}')
print(f'MAPE with key features: {mape_key}')
print(f'ROC AUC with key features: {roc_auc_key}')

# 绘制ROC曲线
fpr_key, tpr_key, _ = roc_curve(y_test_key, y_pred_prob_key)
plt.figure()
plt.plot(fpr_key, tpr_key, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_key:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve with Key Features')
plt.legend(loc="lower right")
plt.show()

