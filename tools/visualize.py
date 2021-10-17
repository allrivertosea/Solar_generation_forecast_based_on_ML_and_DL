import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 该函数用于显示不同模型的误差条形图

def plot_scores(model_names,rmse_train,rmse_test):
    df_score = pd.DataFrame({'model_names' : model_names,
                             'rmse_train' : rmse_train,
                             'rmse_test' : rmse_test})
    df_score = pd.melt(df_score, id_vars=['model_names'], value_vars=['rmse_train', 'rmse_test'])

    plt.figure(figsize=(12, 10))
    sns.barplot(y="model_names", x="value", hue="variable", data=df_score)
    plt.show()

#显示上个月的预测（橙色）和实际值（蓝色）
def plot_predictions(data):
    plt.figure(figsize=(18, 8))
    sns.lineplot(data=data)
    plt.title("Base line predictions (orange) vs real values (blue) for the last month")
    plt.xlabel("hours of the last month (12-2015)")
    plt.ylabel("solar installation efficiency")
    plt.show()

def plot_evolution(X_train,y_train,X_test,y_test,y_pred_test):
    plt.figure(figsize=(18, 8))
    plt.plot(np.arange(len(X_train)), y_train, label='Train')
    plt.plot(np.arange(len(X_train), len(X_train)+len(X_test), 1), y_test, label='Test')
    plt.plot(np.arange(len(X_train), len(X_train)+len(X_test), 1), y_pred_test, label='Test prediction')
    plt.legend()
    plt.show()
