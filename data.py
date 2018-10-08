from sklearn.datasets import load_digits


mnist = load_digits()
X = mnist['data']
y = mnist['target']

X_train, X_test, y_train, y_test = (
    X[:1500], X[1500:], y[:1500], y[1500:]
)
print(X_train.shape)
