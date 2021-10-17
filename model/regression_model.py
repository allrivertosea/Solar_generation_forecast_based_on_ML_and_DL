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
X_train, y_train = train_data[['month', 'week', 'day', 'hour']], train_data['FR10']
X_test, y_test = test_data[['month', 'week', 'day', 'hour']], test_data['FR10']

from sklearn.neighbors import KNeighborsRegressor#k近邻
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet#线性回归，岭回归，Lasso回归，弹性网络
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

import xgboost as xgb


def get_rmse(reg, model_name):
    """打印传入参数的模型的分数以及并返回训练/测试集上的分数"""
    y_train_pred, y_pred = reg.predict(X_train), reg.predict(X_test)
    rmse_train, rmse_test = np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(
        mean_squared_error(y_test, y_pred))
    print(model_name, '\t - RMSE on Training  = {rmse_train:%f}'%rmse_train+' / RMSE on Test = {rmse_test:}'%rmse_test)

    return rmse_train, rmse_test

# 最初使用的所有基本模型的列表
model_list = [
    LinearRegression(), Lasso(), Ridge(), ElasticNet(),
    RandomForestRegressor(), GradientBoostingRegressor(), ExtraTreesRegressor(),
    xgb.XGBRegressor(), KNeighborsRegressor()
             ]
# 训练和测试的分数和名字列表创建
model_names.extend([str(m)[:str(m).index('(')] for m in model_list])


# 训练和测试所有模型s
for model, name in zip(model_list, model_names):
    model.fit(X_train, y_train)
    sc_train, sc_test = get_rmse(model, name)
    rmse_train.append(sc_train)
    rmse_test.append(sc_test)

##支持向量回归不够高效
#
#SVM lin. 	 - RMSE on Training  = 0.31 / RMSE on Test = 0.30
#核支持向量机
#SVM poly. 	 - RMSE on Training  = 0.52 / RMSE on Test = 0.56
##如果我们使用多项式特征将日期时间信息提升到不同的幂
##如果我们将日期时间信息作为分类特征处理，这是处理此类数据的正确方法

