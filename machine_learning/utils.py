import numpy as np

# From Linear Regression, PCA
def mse_func(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

# From Logistic Regression
def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    positive_mask = (z >= 0)
    negative_mask = ~positive_mask

    result = np.empty_like(z)

    # For positive values
    result[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))

    # For negative values
    exp_z = np.exp(z[negative_mask])
    result[negative_mask] = exp_z / (1 + exp_z)

    return result

# Used in KNN, KMeans
# Calculates distance between A (n x d) and B (m x d)
# Returns an (n x m) matrix
def euclidean_distances(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

# Used in PCA
def center(X):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    return X - mean, mean

# Used in Naive Bayes
# Returns n log-likelihoods
def gaussian_logpdf(X, mean, var, eps=1e-9):
    X = np.asarray(X, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64) + eps
    const = -0.5 * np.sum(np.log(2.0 * np.pi * var))
    quad = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
    return const + quad

# Used in Naive-Bayes
def softmax_rows(Z):
    Z = np.asarray(Z, dtype=np.float64)
    Zmax = Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z - Zmax)
    return expZ / expZ.sum(axis=1, keepdims=True)

# Used in Decision Tree
def gini(counts):
    s = counts.sum()
    if s == 0:
        return 0.0
    p = counts / s
    return 1.0 - np.sum(p * p)

# Used in train/test file
def standardize(X, mean=None, std=None):
    if mean is None: mean = X.mean(axis=0)
    if std is None:  std = X.std(axis=0) + 1e-12
    return (X - mean) / std, mean, std # returns (X_std, mean, std)

# Used in multiple models
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(y_true == y_pred)
