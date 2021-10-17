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
# print(df_solar_co.head(2))
# print(df_solar_co.columns)
country_dict = {'AT': 'Austria','BE': 'Belgium','BG': 'Bulgaria','CH': 'Switzerland','CY': 'Cyprus','CZ': 'Czech Republic',
'DE': 'Germany','DK': 'Denmark','EE': 'Estonia','ES': 'Spain','FI': 'Finland','FR': 'France','EL': 'Greece','UK': 'UK','HU': 'Hungary',
'HR': 'Croatia','IE': 'Ireland','IT': 'Italy','LT': 'Lithuania','LU': 'Luxembourg','LV': 'Latvia','NO': 'Norway',
'NL': 'Netherlands','PL': 'Poland','PT': 'Portugal','RO': 'Romania','SE': 'Sweden','SI': 'Slovenia','SK': 'Slovakia'}
# print(len(df_solar_co.columns))#29
df_solar_st = pd.read_csv(path + "\solar_generation_by_station.csv")
# print(df_solar_st.tail(2))
df_solar_st = df_solar_st.drop(columns=['time_step'])#删去该列
# print(len(df_solar_st.columns))#260

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#太阳能站点
df_solar_transposed = df_solar_st[-24*365:].T
df_solar_transposed.tail(2)

#轮廓系数确定簇数
def plot_elbow_scores(df_, cluster_nb):
    km_inertias, km_scores = [], []

    for k in range(2, cluster_nb):
        km = KMeans(n_clusters=k).fit(df_)
        km_inertias.append(km.inertia_)#所有簇平方和
        km_scores.append(silhouette_score(df_, km.labels_))#轮廓系数

    sns.lineplot(range(2, cluster_nb), km_inertias)
    plt.title('elbow graph / inertia depending on k')
    plt.show()

    sns.lineplot(range(2, cluster_nb), km_scores)
    plt.title('scores depending on k')
    plt.show()

plot_elbow_scores(df_solar_transposed, 20)

#对太阳能发电国家
df_solar_transposed = df_solar_co[-24 * 365:].T
plot_elbow_scores(df_solar_transposed, 20)

X = df_solar_transposed

#展示国家聚类结果
km = KMeans(n_clusters=6).fit(X)
X['label'] = km.labels_
print("Cluster nb / Nb of countries in the cluster", X.label.value_counts())

print("Countries grouped by cluster")
for k in range(6):
    print('cluster nb : {%d}'%k', " ".join(list(X[X.label == k].index)))
