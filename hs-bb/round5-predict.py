#coding:utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('round1-4-study-2016.csv')
X = df.drop(['県','県No','学校名','対戦校','得点'], axis=1)
Y = df['得点'].as_matrix()

clf = linear_model.LinearRegression()
clf.fit(X, Y)

# 4回戦の予測
df_next = pd.read_csv('round5-game-predict-2016.csv')
X_next = df_next.drop(['県','県No','学校名','対戦校','予想得点力','結果','差'], axis=1)
next_pred=clf.predict(X_next)

print(next_pred)

