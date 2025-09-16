import numpy as np
from utils import sigmoid

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1)
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return sigmoid(linear)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.int64)
