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
if k.image_data_format() == 'channels_first':
    # 1次元配列に変換
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (img_rows, img_cols, 1)
else:
    # 1次元配列に変換
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 入力データの各画素を0-1の範囲で正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train_shape:', x_train.shape)
# %%
