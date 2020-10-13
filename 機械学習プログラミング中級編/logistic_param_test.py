import warnings
warnings.simplefilter('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# 乳がんのデータセット 
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

model1 = LogisticRegression()
model1.fit(X_train, y_train)
pred = model1.predict(X_test)
print(f'パラメータチューニング前:{accuracy_score(y_test, pred)}')

param_grid = {
    'C':[0.001 * (10 ** i)for i in range(0, 6)],
    'solver':['newton-cg', 'lbfgs', 'libliner', 'sag', 'saga'],
    'warm_start':[True, False]
}
print(param_grid)
grid_search = GridSearchCV(model1, param_grid=param_grid, cv = 5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

best_model = LogisticRegression(C=100, solver='newton-cg', warm_start=True)
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)
print(accuracy_score(y_test, pred))