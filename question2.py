import matplotlib
matplotlib.use('TkAgg')  # 设置使用 TkAgg 后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 设置字体以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号显示为方块的问题

# 尝试不同的编码格式读取数据
try:
    data = pd.read_csv('train.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('train.csv', encoding='GBK')

# 打印列名，检查是否包含预期的列
print("Columns in the dataset:")
print(data.columns)

# 打印前几行数据
print("First few rows of the dataset:")
print(data.head())

# 检查缺失值
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# 填补缺失值（这里简单处理为用均值填充）
data = data.fillna(data.mean())

# 确认数据集中包含的列名
expected_columns = ['id', '洪水概率']
for col in expected_columns:
    if col not in data.columns:
        print(f"Column '{col}' not found in data")
        raise KeyError(f"Column '{col}' not found in data")

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['id', '洪水概率']))

# 添加标准化的洪水概率
flood_probability = data['洪水概率'].values.reshape(-1, 1)
data_scaled = np.hstack((data_scaled, flood_probability))

# 使用KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_scaled)

# 添加聚类结果到原始数据中
data['cluster'] = clusters

# 可视化聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='id', y='洪水概率', hue='cluster', palette='viridis')
plt.title('洪水概率的聚类分析')
plt.savefig('D:/pycharm/数学建模/mathematics modeling/cluster_analysis.png')  # 保存图片文件
plt.show()  # 在 TkAgg 后端显示图像

# 计算各类风险的指标均值
cluster_means = data.groupby('cluster').mean()

# 可视化各类风险的指标特征
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm')
plt.title('各聚类的特征均值')
plt.savefig('D:/pycharm/数学建模/mathematics modeling/feature_means_by_cluster.png')  # 保存图片文件
plt.show()  # 在 TkAgg 后端显示图像

print("Data preprocessing and visualization completed successfully.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 选择更多的特征进行建模
selected_features = ['季风强度', '河流管理', '城市化', '气候变化', '基础设施恶化', '地形排水', '森林砍伐', '大坝质量', '农业实践', '侵蚀', '无效防灾', '排水系统', '海岸脆弱性', '滑坡', '流域', '人口得分', '湿地损失', '规划不足', '政策因素']

# 提取相关特征和目标变量
X = data[selected_features]
y = data['cluster'].astype(int)  # 确保目标变量是整数类型

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 模型评价
print(classification_report(y_test, y_pred))

from imblearn.over_sampling import SMOTE

# 使用SMOTE进行上采样
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 建立随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# 预测
y_pred = model.predict(X_test)

# 模型评价
print(classification_report(y_test, y_pred))


