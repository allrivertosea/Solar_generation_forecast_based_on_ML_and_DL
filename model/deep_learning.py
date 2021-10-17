import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from dataprepare import dataset_con
from visualize import plot_scores,plot_predictions,plot_evolution
import matplotlib.pyplot as plt
import seaborn as sns

##模型训练
model_instances, model_names, rmse_train, rmse_test = [], [], [], []

df = pd.read_csv("dataset\solar_generation_by_station.csv")
df = df[sorted([c for c in df.columns if 'FR' in c])]

# 只保留最近4年的FR数据
df = df[-24*365*4:]
# 数据处理函数：输入为df和lookback，输出的X的各个元素为4年来每个48小时的数据
def process_data(data, past):
    X = []
    for i in range(len(data)-past-1):
        X.append(data.iloc[i:i+past].values)
    return np.array(X)
#根据过去2天的特征值预测之后1个小时的值
lookback = 48
#仅针对FR10这个站点进行预测，y为FR10第一个48小时后的所有数据值，X的元素为y对应的数据值之前的48小时数据
#假设有100行数据，22个特征
# 那么X数组的shape ((100-48+1), 48, 22) ：53行，22列，1166个元素，其中元素shape（48,1），一行有22个
# y具有形状 ((100-48+1), 1) =具有形状 (53, 1)
y = df['FR10'][lookback+1:]
X = process_data(df, lookback)
from sklearn.model_selection import train_test_split
#划分训练集和测试集，不打乱
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

'''
RNN
'''
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Dropout


def my_RNN():
    my_rnn = Sequential()
    my_rnn.add(SimpleRNN(units=32, return_sequences=True, input_shape=(lookback,22)))
    my_rnn.add(SimpleRNN(units=32, return_sequences=True))
    my_rnn.add(SimpleRNN(units=32, return_sequences=False))
    my_rnn.add(Dense(units=1, activation='linear'))
    return my_rnn


rnn_model = my_RNN()
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)

y_pred_train, y_pred_test = rnn_model.predict(X_train), rnn_model.predict(X_test)
err_train_rnn, err_test_rnn = np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))

def append_results(model_name,err_train,err_test):
    model_names.append(model_name)
    rmse_train.append(err_train)
    rmse_test.append(err_test)

append_results("RNN",err_train_rnn,err_test_rnn)


plot_evolution(X_train,y_train,X_test,y_test,y_pred_test)
rnn_res = pd.DataFrame(zip(list(y_test), list(np.squeeze(y_pred_test))), columns =['FR10', 'pred'])
plot_predictions(data=rnn_res[-30*24:])

'''
GRU
'''

from keras.layers import GRU

def my_GRU(input_shape):
    my_GRU = Sequential()
    my_GRU.add(GRU(units=32, return_sequences=True, activation='relu', input_shape=input_shape))
    my_GRU.add(GRU(units=32, activation='relu', return_sequences=False))
    my_GRU.add(Dense(units=1, activation='linear'))
    return my_GRU

gru_model = my_GRU(X.shape[1:])
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

y_pred_train, y_pred_test = gru_model.predict(X_train), gru_model.predict(X_test)
err_train_gru, err_test_gru = np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))

append_results("GRU",err_train_gru,err_test_gru)
plot_evolution(X_train,y_train,X_test,y_test,y_pred_test)

gru_res = pd.DataFrame(zip(list(y_test), list(np.squeeze(y_pred_test))), columns =['FR10', 'pred'])
plot_predictions(data=gru_res[-30*24:])

'''
LSTM
'''

from keras.layers import LSTM

def my_LSTM(input_shape):
    my_LSTM = Sequential()
    my_LSTM.add(LSTM(units=32, return_sequences=True, activation='relu', input_shape=input_shape))
    my_LSTM.add(LSTM(units=32, activation='relu', return_sequences=False))
    my_LSTM.add(Dense(units=1, activation='linear'))
    return my_LSTM

lstm_model = my_LSTM(X.shape[1:])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

y_pred_train, y_pred_test = lstm_model.predict(X_train), lstm_model.predict(X_test)
err_train_lstm, err_test_lstm = np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))
append_results("LSTM",err_train_lstm,err_test_lstm)
plot_evolution(X_train,y_train,X_test,y_test,y_pred_test)

lstm_res = pd.DataFrame(zip(list(y_test), list(np.squeeze(y_pred_test))), columns =['FR10', 'pred'])
plot_predictions(data=lstm_res[-30*24:])

plt.style.use('fivethirtyeight')
plot_scores(model_names,rmse_train,rmse_test)


df_score = pd.DataFrame({'model_names' : model_names, 'rmse_test' : rmse_test})

plt.figure(figsize=(12, 8))
sns.barplot(y="model_names", x="rmse_test", data=df_score, palette="Blues_d")
plt.title("Comparaison des erreurs pour chaque modèle", fontsize=20)
plt.xlabel('erreur RMSE - plus elle est petite, meilleur est le modèle', fontsize=16)
plt.ylabel('liste des modèles esssayés', fontsize=16)
plt.show()
