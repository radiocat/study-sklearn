#coding:utf-8
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('round1-3-study-2016.csv')
X = df.drop(['県','県No','学校名','対戦校','得点'], axis=1)
Y = df['得点'].as_matrix()

clf = linear_model.LinearRegression()
clf.fit(X, Y)

# 3回戦の予測
df_round3 = pd.read_csv('round4-game-predict-2016.csv')
X_round3 = df_round3.drop(['県','県No','学校名','対戦校','予想得点力'], axis=1)
round3_pred=clf.predict(X_round3)

print(round3_pred)

