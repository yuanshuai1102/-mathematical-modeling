import pandas as pd

# 读取Excel文件
file_path = 'data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')  # 修改sheet_name为你需要的工作表名称

# 选择特定的行和列（例如，选择第1到第5行和'A'到'C'列）
selected_rows = df.iloc[23]  # 选择第1到第5行
selected_columns = selected_rows[['A', 'B', 'C']]  # 选择'A', 'B', 'C'列

# 保存为新的Excel文件
new_file_path = 'new_data.xlsx'
selected_columns.to_excel(new_file_path, index=False)

print(f"新表格已保存到 {new_file_path}")

