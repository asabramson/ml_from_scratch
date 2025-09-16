import numpy as np
from utils import euclidean_distances

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng()

        # Pick k random points as centers
        idx = rng.choice(n, size=self.k, replace=False)
        centers = X[idx].copy()

        for _ in range(self.max_iters):
            D = euclidean_distances(X, centers) # (n, k)
            labels = np.argmin(D, axis=1)

            new_centers = centers.copy()
            for j in range(self.k):
                mask = (labels == j)
                if np.any(mask):
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = X[rng.integers(0, n)]

            shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers
            if shift <= self.tol:
                break

        D = euclidean_distances(X, centers)
        labels = np.argmin(D, axis=1)
        self.inertia_ = float(np.sum((X - centers[labels])**2))

        self.centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        D = euclidean_distances(X, self.centers_)
        return np.argmin(D, axis=1)