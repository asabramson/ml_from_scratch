import numpy as np

class SVM:
    def __init__(self, lr=0.01, n_iters=1000, C=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        # Expect 0/1, convert to -1/+1 for hinge loss
        y = np.where(np.asarray(y).reshape(-1) > 0, 1.0, -1.0)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        for _ in range(self.n_iters):
            scores = X @ self.weights + self.bias
            margin = 1.0 - y * scores
            active = margin > 0

            grad_w = self.weights - self.C * (X[active].T @ y[active])
            grad_b = -self.C * np.sum(y[active])

            self.weights -= self.lr * grad_w
            self.bias    -= self.lr * grad_b

    def predict(self, X):
        return (np.asarray(X, dtype=np.float64) @ self.weights + self.bias >= 0.0).astype(np.int64)
