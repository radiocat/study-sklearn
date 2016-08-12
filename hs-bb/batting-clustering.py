#coding:utf-8
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

dataframe = pd.read_csv('batting-2016.csv')

print(dataframe)

del(dataframe['県'])
del(dataframe['学校名'])
del(dataframe['県No'])

array = np.array([dataframe['試合数'].tolist(),
    dataframe['打率'].tolist(),
    dataframe['本塁打'].tolist(),
    dataframe['犠打'].tolist(),
    dataframe['盗塁'].tolist()
    ], np.float)
array = array.T

print(array)

predict = KMeans(n_clusters=5).fit_predict(array)
print(predict)

