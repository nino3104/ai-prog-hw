import pandas as pd
import os

# ----- 1. 加载数据 -----
data_path = "data/yellow_tripdata_2023-01.parquet"
df = pd.read_parquet(data_path)

# ----- 2. 基本信息 -----
print("========== 数据基本信息 ==========")
print(f"数据形状（行数, 列数）：{df.shape}")
print("\n前 5 行数据：")
print(df.head())

print("\n各列数据类型：")
print(df.dtypes)

print("\n缺失值统计：")
print(df.isnull().sum())

print("\n各列基本统计量（数值列）：")
print(df.describe())