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


# ==================== M2 分析可视化 ====================
import matplotlib.pyplot as plt
import matplotlib
import os

# ----- 中文字体设置（防止中文乱码）-----
# 尝试使用 SimHei（Windows 常见中文字体），若无则回退默认
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题
except:
    pass

# 确保 outputs 目录存在
os.makedirs("outputs", exist_ok=True)

# ----- 图1：分小时平均订单量（折线图）-----
print("\n========== 图1：分小时平均订单量 ==========")

# 按小时统计订单数，然后对天数求平均（因为每天订单量波动大，取平均更能体现时间规律）
hourly_orders = df.groupby('pickup_hour').size()
# 计算有多少个不同日期，用于求均值
num_days = df['pickup_date'].nunique()
hourly_avg = hourly_orders / num_days

plt.figure(figsize=(10, 5))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, color='steelblue')
plt.title('2023年1月 纽约市黄色出租车分小时平均订单量', fontsize=14)
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均订单量', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/hourly_avg_orders.png', dpi=150)
plt.show()
print("图1已保存至 outputs/hourly_avg_orders.png")

# ----- 图2：工作日 vs 周末分小时订单量对比 -----
print("\n========== 图2：工作日 vs 周末分小时订单量 ==========")

# 按小时和是否周末分组统计订单数
grouped = df.groupby(['pickup_hour', 'is_weekend']).size().unstack(fill_value=0)
# 分别计算平均（因为工作日和周末天数不同）
weekday_counts = grouped[0] / df[df['is_weekend'] == 0]['pickup_date'].nunique()
weekend_counts = grouped[1] / df[df['is_weekend'] == 1]['pickup_date'].nunique()

plt.figure(figsize=(10, 5))
plt.plot(weekday_counts.index, weekday_counts.values, marker='o', label='工作日', color='steelblue')
plt.plot(weekend_counts.index, weekend_counts.values, marker='s', label='周末', color='darkorange')
plt.title('工作日 vs 周末 分小时平均订单量', fontsize=14)
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均订单量', fontsize=12)
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/hourly_weekday_vs_weekend.png', dpi=150)
plt.show()
print("图2已保存至 outputs/hourly_weekday_vs_weekend.png")

# ----- 图3：区域热度分析（上下客量 TOP10）-----
print("\n========== 图3：区域热度分析 TOP10 ==========")

# 加载区域对照表
zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
# 只保留需要的列，并重命名
zone_names = zone_lookup[['LocationID', 'Zone']].set_index('LocationID')['Zone'].to_dict()

# 统计上车量（PULocationID）TOP10
pu_counts = df['PULocationID'].value_counts().head(10)
# 把数字ID映射为区域名称
pu_labels = [f"{zone_names.get(idx, str(idx))}\n({idx})" for idx in pu_counts.index]

# 统计下车量（DOLocationID）TOP10
do_counts = df['DOLocationID'].value_counts().head(10)
do_labels = [f"{zone_names.get(idx, str(idx))}\n({idx})" for idx in do_counts.index]

# 画图：左右两个子图（上车 vs 下车）
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 上车 TOP10
axes[0].barh(range(10), pu_counts.values[::-1], color='steelblue')
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(pu_labels[::-1], fontsize=9)
axes[0].set_xlabel('订单量', fontsize=12)
axes[0].set_title('上车量 TOP10 区域', fontsize=14)
axes[0].invert_yaxis()  # 让最大的在最上面

# 下车 TOP10
axes[1].barh(range(10), do_counts.values[::-1], color='darkorange')
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(do_labels[::-1], fontsize=9)
axes[1].set_xlabel('订单量', fontsize=12)
axes[1].set_title('下车量 TOP10 区域', fontsize=14)
axes[1].invert_yaxis()

plt.suptitle('2023年1月 NYC黄色出租车 上下客区域热度 TOP10', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('outputs/top10_pu_do_zones.png', dpi=150, bbox_inches='tight')
plt.show()
print("图3已保存至 outputs/top10_pu_do_zones.png")

# ----- 图4（M2-3）：车费影响因素 —— 距离 vs 车费散点图（高峰/非高峰）-----
print("\n========== 图4：车费影响因素 —— 距离 vs 车费 ==========")

# 随机采样 50000 条用于绘图（避免 290 万点导致卡顿）
df_sample = df.sample(n=50000, random_state=42)

plt.figure(figsize=(10, 6))

# 非高峰期（蓝色半透明）
non_peak = df_sample[df_sample['is_peak'] == 0]
plt.scatter(non_peak['trip_distance'], non_peak['fare_amount'], 
            alpha=0.3, s=5, c='steelblue', label='非高峰期')

# 高峰期（橙红色半透明）
peak = df_sample[df_sample['is_peak'] == 1]
plt.scatter(peak['trip_distance'], peak['fare_amount'], 
            alpha=0.3, s=5, c='darkorange', label='高峰期')

plt.title('行程距离 vs 车费（蓝：非高峰期，橙：高峰期）', fontsize=14)
plt.xlabel('行程距离（英里）', fontsize=12)
plt.ylabel('车费（美元）', fontsize=12)
plt.xlim(0, 30)  # 聚焦在主流距离范围，避免极值拉长坐标
plt.ylim(0, 200)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fare_vs_distance_peak.png', dpi=150)
plt.show()
print("图4已保存至 outputs/fare_vs_distance_peak.png")

# ----- 图4补充：乘客人数 vs 车费（箱线图 - 改进版）-----
print("\n========== 图4补充：乘客人数 vs 车费 ==========")

# 只保留常见乘客人数（1~6人）
df_passenger = df[df['passenger_count'].between(1, 6)]

# 按乘客人数分组，构建箱线图数据
passenger_groups = [df_passenger[df_passenger['passenger_count'] == i]['fare_amount'].dropna().values 
                    for i in range(1, 7)]

plt.figure(figsize=(9, 5))
box = plt.boxplot(passenger_groups, tick_labels=range(1, 7), patch_artist=True,
                  medianprops=dict(color='black', linewidth=1.5))

# 给箱子涂上渐变色
colors = ['#c6dbef', '#a6c9e2', '#7fb3d5', '#5b9cc8', '#3b85bb', '#2171b5']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.title('不同乘客人数下的车费分布', fontsize=14)
plt.xlabel('乘客人数', fontsize=12)
plt.ylabel('车费（美元）', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fare_by_passenger.png', dpi=150)
plt.show()
print("图4补充已保存至 outputs/fare_by_passenger.png")


# ----- 图5（M2-4 自选）：各时段平均行程速度（工作日 vs 周末）-----
print("\n========== 图5（自选）：各时段平均行程速度 ==========")

# 过滤掉速度为0或异常的记录（平均速度在合理范围内）
df_speed = df[(df['avg_speed_mph'] > 0) & (df['avg_speed_mph'] < 80)]

# 按小时和是否周末分组，计算平均速度
speed_grouped = df_speed.groupby(['pickup_hour', 'is_weekend'])['avg_speed_mph'].mean().unstack()

plt.figure(figsize=(10, 5))
plt.plot(speed_grouped.index, speed_grouped[0], marker='o', label='工作日', color='steelblue')
plt.plot(speed_grouped.index, speed_grouped[1], marker='s', label='周末', color='darkorange')
plt.title('各时段平均行程速度（工作日 vs 周末）', fontsize=14)
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('平均速度（英里/小时）', fontsize=12)
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/avg_speed_by_hour.png', dpi=150)
plt.show()
print("图5已保存至 outputs/avg_speed_by_hour.png")


# ==================== M3 预测模型 ====================
print("\n========== M3：预测模型 ==========")

# ----- 1. 构建监督学习数据集 -----
# 聚合：按上车区域、小时、星期几 统计订单量
df['pickup_weekday'] = df['pickup_weekday'].astype(int)

agg_cols = ['PULocationID', 'pickup_hour', 'pickup_weekday', 'is_peak', 'is_weekend']
demand_df = df.groupby(agg_cols).size().reset_index(name='demand')

print(f"聚合后样本数：{len(demand_df)}")

# 特征和标签
X = demand_df[['PULocationID', 'pickup_hour', 'pickup_weekday', 'is_peak', 'is_weekend']].values
y = demand_df['demand'].values

# 划分训练/测试集（8:2）
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")


# ----- 2. 神经网络模型（TensorFlow/Keras）-----
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 特征标准化（对区域ID这种大数值很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # 输出：预测需求量（回归任务）
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 训练
history = model.fit(X_train_scaled, y_train, 
                    validation_data=(X_test_scaled, y_test),
                    epochs=50, batch_size=128, verbose=1)

# 画 loss 曲线
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='训练集 loss', color='steelblue')
plt.plot(history.history['val_loss'], label='验证集 loss', color='darkorange')
plt.title('神经网络 Loss 曲线', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/nn_loss_curve.png', dpi=150)
plt.show()
print("神经网络 loss 曲线已保存至 outputs/nn_loss_curve.png")

# 测试集评估
y_pred_nn = model.predict(X_test_scaled).flatten()
mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
print(f"\n神经网络测试集：MAE = {mae_nn:.2f}，RMSE = {rmse_nn:.2f}")


# ----- 3. 随机森林对比模型 -----
print("\n========== 随机森林对比 ==========")

from sklearn.ensemble import RandomForestRegressor

# 随机森林不强制标准化，但标准化了也能用，我们用原始数据保持公平
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"随机森林测试集：MAE = {mae_rf:.2f}，RMSE = {rmse_rf:.2f}")

# 对比汇总
print("\n========== M3 模型对比汇总 ==========")
print(f"{'模型':<12} {'MAE':<12} {'RMSE':<12}")
print(f"{'神经网络':<12} {mae_nn:<12.2f} {rmse_nn:<12.2f}")
print(f"{'随机森林':<12} {mae_rf:<12.2f} {rmse_rf:<12.2f}")
print("\n结论：", end="")
if rmse_rf < rmse_nn:
    print(f"随机森林 RMSE 更低（{rmse_rf:.2f} < {rmse_nn:.2f}），在此任务上表现更优。")
else:
    print(f"神经网络 RMSE 更低（{rmse_nn:.2f} < {rmse_rf:.2f}），在此任务上表现更优。")