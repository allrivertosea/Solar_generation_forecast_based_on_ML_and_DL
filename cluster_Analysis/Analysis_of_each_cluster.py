import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 300

import warnings
warnings.filterwarnings("ignore")
path = "dataset"
df_solar_co = pd.read_csv(path + "\solar_generation_by_country.csv")
df_solar_co = df_solar_co[['NO', 'AT', 'FR', 'FI', 'RO', 'ES']]#以这几个国家代表各自集群
#解决时间戳
def add_time(_df):
    "返回带有两个新列的 DF：分别为时间和小时"
    t = pd.date_range(start='1/1/1986', periods=df_solar_co.shape[0], freq = 'H')
    t = pd.DataFrame(t)
    _df = pd.concat([_df, t], axis=1)
    _df.rename(columns={ _df.columns[-1]: "time" }, inplace = True)
    _df['hour'] = _df['time'].dt.hour
    _df['month'] = _df['time'].dt.month
    _df['week'] = _df['time'].dt.week
    return _df
df_solar_co = add_time(df_solar_co)#转换时间格式
##开始分析
#考虑所有值（包括夜间的）
def plot_hourly(df, title):
    plt.figure(figsize=(14, 9))
    for c in df.columns:
        if c != 'hour':
            sns.lineplot(x="hour", y=c, data=df, label=c)
            #plt.legend(c)
    plt.title(title)
    plt.show()

#一天24小时内各国太阳能发电站的效率
plot_hourly(df_solar_co[df_solar_co.columns.difference(['time', 'month', 'week'])][-24:], "Efficiency of solar stations per country during the 24 hours")
#特定某天各国太阳能发电站的效率
plot_hourly(df_solar_co[df_solar_co.columns.difference(['time', 'month', 'week'])], "Efficiency of solar stations per country during a typical day")
#非空值的站点效率分布（即白天）
temp_df = df_solar_co[df_solar_co.columns.difference(['time', 'hour', 'month', 'week'])]#除列表内容外的所有
plt.figure(figsize=(14, 9))
for col in temp_df.columns:
    sns.distplot(temp_df[temp_df[col] != 0][col], label=col, hist=False)
plt.title("Distribution of the station's efficiency for non null values (ie during the day)")
#法国1985到2015
plt.figure(figsize=(14, 9))
sns.lineplot(x = df_solar_co.time, y = df_solar_co['FR'])
#各国每月的效率
countries = ['NO', 'AT', 'FR', 'FI', 'RO', 'ES']
plt.figure(figsize=(12, 6))
for c in countries:
    temp_df = df_solar_co[[c, 'month']]
    sns.lineplot(x=temp_df["month"], y=temp_df[c], label=c)
plt.xlabel("Month of year")
plt.ylabel("Efficiency")
plt.title("Efficiency across the months per country")
# 各国每周的效率
plt.figure(figsize=(12, 6))
for c in countries:
    temp_df = df_solar_co[[c, 'week']]
    sns.lineplot(x=temp_df["week"], y=temp_df[c], label=c)
plt.xlabel("Week of year")
plt.ylabel("Efficiency")
plt.title("Efficiency across the weeks per country")
temp_df = df_solar_co.copy()
temp_df['year'] = temp_df['time'].dt.year

plt.figure(figsize=(12, 6))
for c in countries:
    temp_df_ = temp_df[[c, 'year']]
    sns.lineplot(x=temp_df_["year"], y=temp_df_[c], label=c)
plt.xlabel("Year")
plt.ylabel("Efficiency")
plt.title("Efficiency across the years per country")

#各国第三四分位
temp_df = df_solar_co[(5 < df_solar_co.hour) & (df_solar_co.hour < 22)]
temp_df = temp_df.drop(columns=['time', 'hour', 'month', 'week'])
temp_df.describe()

def plot_by_country(_df, title, nb_col):
    _df = _df.describe().iloc[nb_col, :]
    plt.figure(figsize=(14, 6))
    sns.barplot(x=_df.index, y=_df.values)
    plt.title(title)
    plt.show()

plot_by_country(temp_df,"Mean efficiency by country", 1)
#小提琴图了解密度
plt.figure(figsize=(14, 8))
sns.violinplot(x="variable", y="value", data=pd.melt(temp_df))
#箱线图
plt.figure(figsize=(14, 8))
sns.boxplot(x="variable", y="value", data=pd.melt(temp_df))
#站点的效率分布
plt.figure(figsize=(14, 9))
for col in temp_df.columns:
    sns.distplot(temp_df[temp_df[col] != 0][col], label=col, hist=False)
plt.title("Distribution of the station's efficiency")

#相关图
def plot_corr(df_):
    corr = df_.corr()

    # 为上三角形生成掩码
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # 设置 matplotlib 图
    f, ax = plt.subplots(figsize=(14, 12))

    # 绘制热图并修正纵横比
    sns.heatmap(corr, mask=mask, center=0, square=True, cmap='Spectral', linewidths=.5, cbar_kws={"shrink": .5})

plot_corr(temp_df)

temp_df.corr()

#热力地图：月份和时

df_solar_co['year'] = df_solar_co['time'].dt.year
plt.figure(figsize=(8, 6))
temp_df = df_solar_co[['FR', 'month', 'hour']]
temp_df = temp_df.groupby(['hour', 'month']).mean()
temp_df = temp_df.unstack('month').sort_index(ascending=False)
sns.heatmap(temp_df, vmin = 0.09, vmax = 0.29, cmap = 'plasma')

plt.show()
