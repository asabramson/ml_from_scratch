import numpy as np
from utils import center

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc, mu = center(X)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        k = X.shape[1] if self.n_components is None else int(self.n_components)
        k = max(1, min(k, Vt.shape[0]))

        self.components_ = Vt[:k] # top-k principal axes
        self.mean_ = mu
        self.singular_values_ = S[:k]

        # Variance along each PC
        n = X.shape[0]
        var = (S**2) / (n - 1) if n > 1 else np.zeros_like(S)
        total_var = var.sum() or 1.0
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_
