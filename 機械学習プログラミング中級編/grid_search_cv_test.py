from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# アイリスデータセットを読み込む
datasets = datasets.load_iris()

X = datasets.data
y = datasets.target

model = DecisionTreeClassifier()

# 試行するパラメータの羅列
params = {
    'max_depth': [i for i in range(1, 20)],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(model, param_grid=params, cv=50)
grid_search.fit(X, y)

# 最も良かったスコア
print(grid_search.best_score_)
# 最適なモデル
print(grid_search.best_estimator_)
# 最適なパラメータの組み合わせ
print(grid_search.best_params_)
