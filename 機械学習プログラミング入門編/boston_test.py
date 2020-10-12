# ボストンの住宅価格データセットの読み込み
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target

# ライブラリの読み込み
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データフレームの作成
df_x = pd.DataFrame(X)
df_y = pd.DataFrame(y)
df_x.columns = boston.feature_names
df_y.columns = ['target']
df = pd.concat([df_x, df_y], axis=1)
# print(df.head())

#相関係数の算出
corr = df.corr()
# print(corr)

# 相関係数の算出からLSTATの相関が強いので、LSTATで単回帰分析を行う。
lstat = df_x.loc[:, 'LSTAT']
lstat = lstat.values
lstat = lstat.reshape(-1,1)
# print(lstat)

# データを訓練用とテスト用に分ける
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(lstat, y, test_size=0.3, random_state=0)

# 線形モデルのインスタンス作成
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# モデルの学習
lr.fit(X_train, y_train)

# モデルの可視化
# plt.scatter(lstat, y)
# plt.plot(X_test, lr.predict(X_test), color='red')
# plt.title('boston_housing')
# plt.xlabel('LSTAT')
# plt.ylabel('target')
# plt.show()

# モデルの評価
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

import math
from sklearn.metrics import mean_squared_error, r2_score
print(f'RMSE : {math.sqrt(mean_squared_error(y_test, y_test_pred))}')
print(f'R2 Train : {r2_score(y_train, y_train_pred)}, R2 Test : {r2_score(y_test, y_test_pred)}')