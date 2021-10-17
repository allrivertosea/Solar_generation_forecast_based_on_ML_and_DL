import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 300

import warnings
warnings.filterwarnings("ignore")
##数据准备
df = pd.read_csv("dataset\solar_generation_by_station.csv")

def add_date_time(_df):
    "返回带有两个新列的 DF：一天中的时间和小时"
    t = pd.date_range(start='1/1/1986', periods=_df.shape[0], freq = 'H')
    t = pd.DataFrame(t)
    _df = pd.concat([_df, t], axis=1)
    _df.rename(columns={ _df.columns[-1]: "time" }, inplace = True)
    _df['year'] = _df['time'].dt.year
    _df['month'] = _df['time'].dt.month
    _df['week'] = _df['time'].dt.weekofyear
    _df['day'] = _df['time'].dt.dayofyear
    _df['hour'] = _df['time'].dt.hour
    return _df
def dataset_con(df):
    #只保留几年和我们感兴趣的列
    df = add_date_time(df)
    df = df[~df.year.isin([1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016])]
    # 只保留要用于预测的国家的数值
    df = df[['FR10', 'year', 'month', 'week', 'day', 'hour', 'time']]

    #划分数据集，训练集和测试集
    # train data 10年
    train_data = df[-24*365*10:-24*31]
    # test data 2015最后一个月的记录
    test_data = df[-24*31:]
    return train_data,test_data
