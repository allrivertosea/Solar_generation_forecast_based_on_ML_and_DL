import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from dataprepare import dataset_con
from visualize import plot_scores,plot_predictions
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 300

##原始数据
df = pd.read_csv("F:\mygithub\Big_Data_Renewable_energies-master\dataset\solar_generation_by_station.csv")
train_data,test_data = dataset_con(df)

##模型训练

model_instances, model_names, rmse_train, rmse_test = [], [], [], []

#构造训练集和测试集
x_train, y_train = train_data.drop(columns=['time']), train_data['FR10']
x_test, y_test = test_data.drop(columns=['time']), test_data['FR10']

# 基线模型，作为基准模型
def mean_df(d, h):
    "return the hourly mean of a specific day of the year"
    res = x_train[(x_train['day'] == d) & (x_train['hour'] == h)]['FR10'].mean()
    return res
#预测值添加到数据集
x_train['pred'] = x_train.apply(lambda x: mean_df(x.day, x.hour), axis=1)
x_test['pred'] = x_test.apply(lambda x: mean_df(x.day, x.hour), axis=1)
model_names.append("base_line")
rmse_train.append(np.sqrt(mean_squared_error(x_train['FR10'], x_train['FR10']))) # a modifier en pred
rmse_test.append(np.sqrt(mean_squared_error(x_test['FR10'], x_test['pred'])))
#显示上个月的预测（橙色）和实际值（蓝色）
plot_predictions(data=x_test[['FR10', 'pred']])
