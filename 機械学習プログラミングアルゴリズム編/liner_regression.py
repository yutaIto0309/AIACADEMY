# ピザの直径
x = [[12], [16], [20], [28], [36]]
# ピザの値段
y = [[700], [900], [1300], [1750], [1800]]

# ピザの直径と値段のグラフの描画
# import matplotlib.pyplot as plt
# plt.figure()
# plt.title('Relation between diameter and price') # タイトル
# plt.xlabel('diameter') # X軸ラベル
# plt.ylabel('price') # Y軸ラベル
# plt.scatter(x, y) # 散布図の作成
# plt.axis([0, 50, 0, 2500]) # 表の最小値
# plt.grid(True)
# plt.show()

# 単回帰分析のモデルで学習
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)

# 25cmのピザの値段を予測
import numpy as np
price = model.predict(np.array([25]).reshape(-1,1))
print(f'25 cm pizza should cost : ${price[0][0]}')

# テストデータの作成
X_test = [[16], [18], [22], [32], [24]]
y_test = [[1100], [850], [1500], [1800], [1100]]

score = model.score(X_test, y_test)
print(f'r-squared:{score}')