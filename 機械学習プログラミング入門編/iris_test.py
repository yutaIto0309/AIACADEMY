from sklearn import datasets
from sklearn import svm

#アヤメのデータを読み込む
iris = datasets.load_iris()
# print(iris.data)
# print(iris.data.shape)

num = len(iris.data)
# print(num)

clf = svm.LinearSVC()
clf.fit(iris.data, iris.target)
print(clf.predict([[1.4, 3.5, 5.1, 0.2], [6.5, 2.6, 4.4, 1.4], [5.9, 3.0, 5.2, 1.5]]))