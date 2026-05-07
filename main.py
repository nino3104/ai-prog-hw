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

# ----- 4. 数据清洗 -----
print("\n========== 开始数据清洗 ==========")

# 记录清洗前总行数
before = len(df)

# 4.1 移除上车时间不在2023年1月内的记录（48条）
df = df[(df['tpep_pickup_datetime'] >= '2023-01-01') & 
        (df['tpep_pickup_datetime'] < '2023-02-01')]
print(f"移除时间异常记录后：{len(df)} 条（移除 {before - len(df)} 条）")

# 4.2 移除行程距离异常（<=0 或 >100 英里）
df = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= 100)]
print(f"移除距离异常记录后：{len(df)} 条（移除 {before - len(df)} 条累计）")

# 4.3 移除车费或总金额为负的记录
df = df[(df['fare_amount'] >= 0) & (df['total_amount'] >= 0)]
print(f"移除费用负值记录后：{len(df)} 条（累计移除 {before - len(df)} 条）")

# 4.4 处理乘客数异常（0 或 >5）—— 修正为 1（最常出现的值），而不是整行删除
df.loc[(df['passenger_count'] == 0) | (df['passenger_count'] > 5), 'passenger_count'] = 1
print(f"修正乘客数异常条数：{len(df[(df['passenger_count'] == 0) | (df['passenger_count'] > 5)])}")

# 4.5 处理五列套餐缺失 —— 直接删除这些行（只有 2.34%，对模型影响极小）
before_drop = len(df)
df = df.dropna(subset=['passenger_count', 'RatecodeID', 'store_and_fwd_flag', 
                       'congestion_surcharge', 'airport_fee'])
print(f"移除套餐缺失行后：{len(df)} 条（移除 {before_drop - len(df)} 条）")

print(f"清洗完成！最终数据量：{len(df)} 条，共移除 {before - len(df)} 条 ({((before - len(df))/before)*100:.2f}%)")

# ----- 5. 特征提取 -----
print("\n========== 开始特征提取 ==========")

# 5.1 从 pickup 时间中提取小时、星期几、日期
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour          # 小时（0-23）
df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday    # 星期几（0=周一, 6=周日）
df['pickup_date'] = df['tpep_pickup_datetime'].dt.date          # 日期（方便按天聚合）

# 5.2 是否高峰期（早高峰 7-9，晚高峰 17-19）
df['is_peak'] = df['pickup_hour'].isin([7,8,9,17,18,19]).astype(int)

# 5.3 是否周末（5=周六, 6=周日）
df['is_weekend'] = df['pickup_weekday'].isin([5,6]).astype(int)

# 5.4 行程时长（分钟）
df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

# 5.5 衍生特征1：平均速度（英里/小时）—— 行程距离除以行程时长
df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration_min'] / 60)
# 过滤速度异常值（0或超大值，后续若产生inf由外部清洗处理）

# 5.6 衍生特征2：每英里车费（fare_amount / trip_distance）
df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']

# 打印新特征统计，确认提取成功
print("新增特征：")
print(df[['pickup_hour', 'pickup_weekday', 'is_peak', 'is_weekend', 
          'trip_duration_min', 'avg_speed_mph', 'fare_per_mile']].head())
print(f"\n最终数据量：{len(df)} 条，共 {len(df.columns)} 列")