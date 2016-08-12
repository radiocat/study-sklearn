#coding:utf-8
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

batting_dataframe = pd.read_csv('batting-2016.csv')
pitching_dataframe = pd.read_csv('pitching-2016.csv')

array = np.array([batting_dataframe['試合数'].tolist(),
    batting_dataframe['打率'].tolist(),
    batting_dataframe['本塁打'].tolist(),
    batting_dataframe['犠打'].tolist(),
    batting_dataframe['盗塁'].tolist(),
    pitching_dataframe['投手数'].tolist(),
    pitching_dataframe['投球回数'].tolist(),
    pitching_dataframe['失点'].tolist(),
    pitching_dataframe['防御率'].tolist()
    ], np.float)
array = array.T

print(array)

predict = KMeans(n_clusters=5).fit_predict(array)
print(predict)



