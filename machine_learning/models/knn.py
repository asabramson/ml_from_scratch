import numpy as np
from utils import euclidean_distances
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.int64)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        D = euclidean_distances(X, self._X)
        idx = np.argpartition(D, self.k, axis=1)[:, :self.k] # k nearest indices
        votes = self._y[idx]
        preds = np.empty(votes.shape[0], dtype=np.int64)
        for i, row in enumerate(votes):
            counts = np.bincount(row)
            preds[i] = counts.argmax()
        return preds
