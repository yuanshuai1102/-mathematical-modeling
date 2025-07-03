import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the file
file_path = 'submit.csv'
data = pd.read_csv(file_path)

# Sampling the data (e.g., every 100th point)
sampled_data = data[::100]

# Plotting the sampled data
plt.figure(figsize=(10, 6))
plt.plot(sampled_data['id'], sampled_data['洪水概率'], marker='o', linestyle='-')
plt.xlabel('ID')
plt.ylabel('洪水概率')
plt.title('洪水概率随ID变化的折线图（采样后）')
plt.grid(True)
plt.show()
