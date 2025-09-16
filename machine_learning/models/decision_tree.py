import numpy as np
from utils import gini

class _Node:
    __slots__ = ("leaf", "pred", "proba", "feat", "thr", "L", "R")
    def __init__(self, leaf, pred=None, proba=None, feat=None, thr=None, L=None, R=None):
        self.leaf = leaf
        self.pred = pred
        self.proba = proba
        self.feat = feat
        self.thr = thr
        self.L = L
        self.R = R

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth # None = unlimted depth
        self.min_samples_split = max(2, int(min_samples_split)) # minimum number of samples to consider a split
        self.min_impurity_decrease = float(min_impurity_decrease) # require at least this impurity to split
        self.root_ = None
        self.classes_ = None
        self.n_classes_ = None # length of classes

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        self.root_ = self._grow(X, y_idx, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=int)
        for i, row in enumerate(X):
            out[i] = self._predict_row(self.root_, row)
        return self.classes_[out]
    

    def _grow(self, X, y, depth):
        n, d = X.shape
        counts = np.bincount(y, minlength=self.n_classes_)
        pred = int(np.argmax(counts))
        proba = counts / counts.sum()

        # Stop if either:
        # 1. Pure 
        # 2. Shallow sample
        # 3. Depth limit
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n < self.min_samples_split or \
           counts.max() == n:
            return _Node(True, pred=pred, proba=proba)

        parent_imp = gini(counts)

        # Find best split across features
        best_gain, best_feat, best_thr, best_mask = 0.0, None, None, None
        for j in range(d):
            gain, thr, left_mask = self._best_split(X[:, j], y, parent_imp)
            if gain > best_gain:
                best_gain, best_feat, best_thr, best_mask = gain, j, thr, left_mask

        if best_feat is None or best_gain < self.min_impurity_decrease:
            return _Node(True, pred=pred, proba=proba)

        L = best_mask
        R = ~L
        left = self._grow(X[L], y[L], depth + 1)
        right = self._grow(X[R], y[R], depth + 1)
        return _Node(False, feat=best_feat, thr=best_thr, L=left, R=right)

    def _best_split(self, xj, y, parent_imp):
        # Sort by feature
        order = np.argsort(xj, kind="mergesort")
        x = xj[order]
        y = y[order]

        if x[0] == x[-1]:
            return 0.0, None, None  # no split possible

        C = self.n_classes_
        n = len(y)

        left_counts = np.zeros((n, C), dtype=int)
        left_counts[np.arange(n), y] = 1
        left_counts = left_counts.cumsum(axis=0)
        total = left_counts[-1].copy()

        best_gain, best_thr = 0.0, None
        best_idx = None

        for i in range(n - 1):
            # Consider thresholds only where feature value changes
            if x[i] == x[i + 1]:
                continue

            lc = left_counts[i]
            rc = total - lc
            nl = lc.sum()
            nr = rc.sum()
            if nl == 0 or nr == 0:
                continue

            imp_l = gini(lc)
            imp_r = gini(rc)
            child_imp = (nl * imp_l + nr * imp_r) / (nl + nr)
            gain = parent_imp - child_imp

            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_thr = (x[i] + x[i + 1]) / 2.0

        if best_idx is None:
            return 0.0, None, None

        left_mask = xj <= best_thr
        return best_gain, best_thr, left_mask

    def _predict_row(self, node, row):
        while not node.leaf:
            node = node.L if row[node.feat] <= node.thr else node.R
        return node.pred
