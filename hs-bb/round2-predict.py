#coding:utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('round1-result-2016.csv')
X = df.drop(['県','県No','学校名','対戦校','得点'], axis=1)
Y = df['得点'].as_matrix()

clf = linear_model.LinearRegression()
clf.fit(X, Y)

# 2回戦の予測
df_round2 = pd.read_csv('round2-game-2016.csv')
X_round2 = df_round2.drop(['県','県No','学校名','対戦校'], axis=1)
round2_pred=clf.predict(X_round2)

print(round2_pred)

