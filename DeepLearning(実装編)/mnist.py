#%%
import keras
from keras.datasets import mnist
from keras import   Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt 

"""
mnist : 手書き数字画像データセット
Sequential : Kerasを用いてモデルを生成するためのモジュール
Dense : 全結合増のレイヤモジュール
Dropout : ドロップアウトモジュール
Conv2D : 2次元畳み込み層のモジュール
MaxPool2D : 2次元最大プーリング層のモジュール
Flatten : 入力を平滑化するモジュール
"""

batch_size = 128
num_classes = 10
epochs = 12

# 入力画像の大きさ
img_rows, img_cols = 28, 28

# 学習データとテストデータに分割したデータ
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
# %%
