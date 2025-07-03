import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('/mnt/data/city_data.csv')

# 定义评价指标
criteria = ['GDP', '人口', 'AQI', '绿化覆盖率', '文化遗产数量', '博物馆数量', '交通便捷度', '气温', '降水量', '美食评分']

# 数据标准化处理
def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

for criterion in criteria:
    data[f'{criterion}_norm'] = normalize(data[criterion])

# 标准化后的数据矩阵
norm_data = data[[f'{criterion}_norm' for criterion in criteria]].values

# 计算各指标的熵值
epsilon = 1e-10  # 防止出现log(0)
P = norm_data / norm_data.sum(axis=0)
E = -np.nansum(P * np.log(P + epsilon), axis=0) / np.log(len(data))

# 计算各指标的权重
d = 1 - E
weights = d / d.sum()

# 输出权重
weights_dict = dict(zip(criteria, weights))
print('指标权重:', weights_dict)

# 计算综合评分
data['综合评分'] = np.dot(norm_data, weights)

# 按照综合评分进行降序排序，并选出前50个城市
top_50_cities = data.sort_values(by='综合评分', ascending=False).head(50)

# 输出结果
print(top_50_cities[['城市', '综合评分']])

