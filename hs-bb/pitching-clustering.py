#coding:utf-8
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

dataframe = pd.read_csv('pitching-2016.csv')

print(dataframe)

del(dataframe['県'])
del(dataframe['学校名'])
del(dataframe['県No'])
del(dataframe['主戦投手'])

array = np.array([dataframe['投手数'].tolist(),
    dataframe['投球回数'].tolist(),
    dataframe['失点'].tolist(),
    dataframe['防御率'].tolist()
    ], np.float)
array = array.T

print(array)

predict = KMeans(n_clusters=5).fit_predict(array)
print(predict)


