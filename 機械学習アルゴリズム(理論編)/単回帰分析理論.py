import numpy as np
import matplotlib.pyplot as plt 

train_xy = np.loadtxt("/Users/itouyuuta/Desktop/Python/pythonStudies/AIAcademy/機械学習アルゴリズム(理論編)/download_cost.csv", delimiter=',', skiprows=1)
train_x = train_xy[:, 0]
train_y = train_xy[:,1]

# データをプロットする
# plt.plot(train_x, train_y, 'o')
# plt.title('download per cost')
# plt.xlabel('cost')
# plt.ylabel('download')
# plt.show()

theta0 = np.random.rand()
theta1 = np.random.rand()


def f(x, b, a):
    return b + a*x

def E(x, y, b, a):
    return (1/2) * np.sum((y - f(x, b, a))**2)

mu_x = train_x.mean()
sigma_x = train_x.std()

mu_y = train_y.mean()
sigma_y = train_y.mean()

def standard_scale(a, mu, sigma):
    return (a - mu) / sigma

standard_x = standard_scale(train_x, mu_x, sigma_x)
standard_y = standard_scale(train_y, mu_y, sigma_y)

# plt.plot(standard_x, standard_y, 'o')
# plt.show()

ETA = 0.001
diff = 1
count = 0

error = E(standard_x, standard_y, theta0, theta1)

while abs(diff) > 0.0001:
    theta0 = theta0 - (ETA * np.sum(f(standard_x, theta0, theta1) - standard_y))
    theta1 = theta1 - (ETA * np.sum((f(standard_x, theta0, theta1) - standard_y)* standard_x)) 

    current_error = E(standard_x, standard_y, theta0, theta1)
    diff = error - current_error
    error = current_error
    count += 1

# 結果の出力
print(f'切片:{theta0}')
print(f'係数:{theta1}')