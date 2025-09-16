import numpy as np
from models.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=40, max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0, bootstrap=True):
        self.n_trees = int(n_trees)
        self.max_depth = max_depth # max depth of each tree
        self.min_samples_split = int(min_samples_split) # minimum number of samples to consider a split
        self.min_impurity_decrease = float(min_impurity_decrease) # require at least this impurity to split
        self.bootstrap = bool(bootstrap) # true by default, draw bootstrap samples for each tree (a single data point may be included multiple times in a single bootstrap sample, and some data points may not be included at all)

        self.trees_ = []
        self.features_ = []
        self.classes_ = None
        self.rng_ = np.random.default_rng()

    def _choose_features(self, d):
        k = max(1, int(np.sqrt(d)))
        return np.sort(self.rng_.choice(d, size=k, replace=False))

    def _bootstrap_indices(self, n):
        if not self.bootstrap:
            return np.arange(n)
        return self.rng_.integers(0, n, size=n)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        n, d = X.shape

        self.classes_, y_idx = np.unique(y, return_inverse=True)

        self.trees_.clear()
        self.features_.clear()

        for _ in range(self.n_trees):
            feat_idx = self._choose_features(d)
            row_idx = self._bootstrap_indices(n)

            X_sub = X[row_idx][:, feat_idx]
            y_sub = y_idx[row_idx]

            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_impurity_decrease=self.min_impurity_decrease)
            tree.fit(X_sub, y_sub)
            self.trees_.append(tree)
            self.features_.append(feat_idx)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Collect predictions from each tree
        all_preds = np.empty((self.n_trees, n), dtype=int)
        for t, (tree, feat_idx) in enumerate(zip(self.trees_, self.features_)):
            all_preds[t] = tree.predict(X[:, feat_idx])

        # Majority vote per sample
        final = np.empty(n, dtype=int)
        for i in range(n):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            final[i] = vals[np.argmax(counts)]
        return self.classes_[final]