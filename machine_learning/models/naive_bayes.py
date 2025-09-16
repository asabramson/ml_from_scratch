import numpy as np
from utils import gaussian_logpdf, softmax_rows

class NaiveBayes:
    def __init__(self):
        self.classes_ = None
        self.class_prior_log_ = None
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        C = self.classes_.size
        d = X.shape[1]

        self.mean_ = np.zeros((C, d), dtype=np.float64)
        self.var_  = np.zeros((C, d), dtype=np.float64)
        self.class_prior_log_ = np.zeros(C, dtype=np.float64)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.mean_[i] = Xc.mean(axis=0)
            self.var_[i]  = Xc.var(axis=0)
            self.class_prior_log_[i] = np.log(len(Xc) / len(X))
        return self

    def _joint_log_likelihood(self, X):
        ll = []
        for i in range(len(self.classes_)):
            ll.append(self.class_prior_log_[i] + gaussian_logpdf(X, self.mean_[i], self.var_[i]))
        return np.column_stack(ll)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    # def predict_proba(self, X):
    #     X = np.asarray(X, dtype=np.float64)
    #     jll = self._joint_log_likelihood(X)
    #     return softmax_rows(jll)