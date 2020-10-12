from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()

# 画像データの表示
# img = np.reshape(digits.data[0], (8,8))
# plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
# plt.axis('off')
# plt.show()

# 画像とラベルの表示
# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:10]):
#     plt.subplot(2, 5, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title(f'Training: {label}')
# plt.show()

# 学習データとテストデータに分ける
num = len(digits.data)
training_num = num * 2 // 3
# 学習データの生成
training_data =  digits.data[:training_num]
training_target = digits.target[:training_num]
# テストデータの生成
test_data = digits.data[training_num:]
test_target = digits.target[training_num:]

#svmモデルの作成
classifier = svm.SVC(gamma=0.001)
classifier.fit(training_data, training_target)

#作成したモデルで予測
predicted = classifier.predict(test_data)

image_and_predictions = list(zip(digits.images[training_num:], predicted))
for index, (image, prediction) in enumerate(image_and_predictions[:20]):
    plt.subplot(4, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Training: {prediction}')
plt.show()

#正解率の表示
metrics.accuracy_score(test_target, predicted)
