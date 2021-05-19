from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
X, y = load_boston(True)
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
ridge=Ridge()
y_pred_ridge = ridge.fit(X_train, y_train).predict(X_test)
r2_score_ridge = r2_score(y_test, y_pred_ridge)
print(r2_score_ridge)
