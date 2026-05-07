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

# ----- 3. 数据质量报告（扩展） -----
print("\n========== 数据质量报告 ==========")

# 缺失率（以百分比显示，保留2位小数）
missing_count = df.isnull().sum()
missing_rate = (missing_count / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    '缺失数量': missing_count,
    '缺失率(%)': missing_rate
})
print("\n各列缺失情况：")
print(missing_report[missing_report['缺失数量'] > 0])  # 只显示有缺失的列

# 异常值统计（基于常见业务规则）
print("\n异常值统计：")

# 1. 行程距离异常（<=0 或 >100 英里）
abnormal_distance = df[(df['trip_distance'] <= 0) | (df['trip_distance'] > 100)]
print(f"行程距离异常（<=0 或 >100 英里）：{len(abnormal_distance)} 条")

# 2. 乘客数异常（为 0 或 >5）
abnormal_passenger = df[(df['passenger_count'] == 0) | (df['passenger_count'] > 5)]
print(f"乘客数异常（0 或 >5）：{len(abnormal_passenger)} 条")

# 3. 费用相关负值异常
negative_fare = df[df['fare_amount'] < 0]
print(f"车费为负值：{len(negative_fare)} 条")
negative_total = df[df['total_amount'] < 0]
print(f"总金额为负值：{len(negative_total)} 条")

# 4. 时间异常（pickup 早于 2023-01-01 或晚于 2023-01-31）
abnormal_time = df[(df['tpep_pickup_datetime'] < '2023-01-01') | 
                   (df['tpep_pickup_datetime'] >= '2023-02-01')]
print(f"上车时间不在2023年1月内：{len(abnormal_time)} 条")

# 5. 经纬度区域异常（PULocationID 或 DOLocationID 为 0，0 通常无效）
abnormal_pu = df[df['PULocationID'] == 0]
abnormal_do = df[df['DOLocationID'] == 0]
print(f"上车区域ID为0：{len(abnormal_pu)} 条")
print(f"下车区域ID为0：{len(abnormal_do)} 条")

print("\n数据质量报告生成完毕。")