import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体路径
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 使用黑体字体，确保路径正确
# 创建字体对象
font = FontProperties(fname=font_path)
# Load the files with appropriate encoding
train_df = pd.read_csv('train.csv', encoding='latin1')
test_df = pd.read_csv('test1.csv', encoding='latin1')

# Extract the relevant columns
train_prob = train_df.iloc[:10, [-1]]
test_prob = test_df.iloc[:10, [-1]]

# Rename columns for clarity
train_prob.columns = ['洪水概率']
test_prob.columns = ['洪水概率']


# Plot the line charts
plt.figure(figsize=(10, 6))
plt.plot(train_prob, label='实际',marker='o')
plt.plot(test_prob, label='预测',marker='x')
plt.title('预测对比', fontproperties=font)
plt.xlabel('序号', fontproperties=font)
plt.ylabel('洪水概率', fontproperties=font)
plt.legend(prop=font)
plt.grid(True)

# 设置y轴刻度，范围从0.35到0.6，间隔为0.05
yticks = [i * 0.1 for i in range(2, 6)]  # 0.35, 0.40, 0.45, ..., 0.60
plt.yticks(yticks, [f'{tick:.2f}' for tick in yticks], fontproperties=font)

plt.show()
