"""
Models implemented from scratch with NumPy (as in the paper):
  * GaussianNaiveBayes      - generative model, strong feature-independence assumption
  * LogisticRegressionNewton - discriminative model trained with Newton's method
"""
import numpy as np


class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class GaussianNaiveBayes:
    """P(y|x) ∝ P(y) * prod_j N(x_j | mu_{y,j}, var_{y,j})"""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.array([0, 1])
        eps = self.var_smoothing * X.var(axis=0).max()
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.mean_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) + eps for c in self.classes_])
        return self

    def _log_joint(self, X):
        X = np.asarray(X, float)
        out = []
        for i in range(len(self.classes_)):
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[i]) + (X - self.mean_[i]) ** 2 / self.var_[i],
                axis=1,
            )
            out.append(np.log(self.prior_[i]) + log_lik)
        return np.column_stack(out)

    def predict_proba(self, X):
        lj = self._log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)  # numerical stability
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importance(self):
        """Standardized gap between class means: |mu1 - mu0| / pooled std."""
        pooled = np.sqrt((self.var_[0] + self.var_[1]) / 2)
        return np.abs(self.mean_[1] - self.mean_[0]) / pooled


