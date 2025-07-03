import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = 'City_data.xlsx'
data = pd.read_excel(file_path)

# 选取需要的指标列
columns = ['AQI', '绿化覆盖率 (%)', '废水处理率 (%)', '废气处理率 (%)', '垃圾分类处理率 (%)', '历史遗迹数量', '博物馆数量', '文化活动频次', '文化设施数量', '公共交通覆盖率 (%)', '线路密度 (km/km²)', '高速公路里程 (km)', '机场航班数量', '年平均气温 (℃)', '年降水量 (mm)', '适宜旅游天数', '空气湿度 (%)', '餐馆数量', '特色美食数量', '美食活动频次']
cities = data['市']
data = data[columns]

# 数据标准化
data_std = (data - data.min()) / (data.max() - data.min())

# 熵权法计算权重
P = data_std / data_std.sum(axis=0)
E = -np.sum(P * np.log(P + 1e-9), axis=0) / np.log(len(data))
D = 1 - E
W = D / D.sum()

# TOPSIS法计算综合得分
V = data_std * W
ideal_solution = np.max(V, axis=0)
negative_ideal_solution = np.min(V, axis=0)
distance_to_ideal = np.sqrt(np.sum((V - ideal_solution)**2, axis=1))
distance_to_negative_ideal = np.sqrt(np.sum((V - negative_ideal_solution)**2, axis=1))
scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

# 将综合得分添加到数据框中
data['Score'] = scores
data['City'] = cities

# 按照综合得分排序，选出前50个城市
top_50_cities = data.sort_values(by='Score', ascending=False).head(50)

# 输出前50个城市及其分数
top_50_cities_list = top_50_cities[['City', 'Score']].values.tolist()
print(top_50_cities_list)

# 生成柱形图
plt.figure(figsize=(12, 8))
plt.barh(top_50_cities['City'], top_50_cities['Score'], color='skyblue')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
plt.xlabel('综合得分')
plt.ylabel('城市')
plt.title('最令外国游客向往的50个城市')
plt.gca().invert_yaxis()  # 反转Y轴，使得分最高的城市在最上面
plt.show()

# 输出前50个城市
top_50_cities_list = top_50_cities['City'].tolist()
print(top_50_cities_list)


