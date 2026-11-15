
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

class LogisticRegressionNewton:
    """Binary logistic regression fit with Newton-Raphson (IRLS).

    Features are standardized internally. A small L2 penalty (l2) keeps the
    Hessian invertible and stops weights diverging on (near-)separable data.
    """

    def __init__(self, l2: float = 1e-2, max_iter: int = 50, tol: float = 1e-8):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.scaler_ = StandardScaler()
        Xb = np.hstack([np.ones((len(X), 1)), self.scaler_.fit_transform(X)])
        n, d = Xb.shape
        reg = self.l2 * np.eye(d)
        reg[0, 0] = 0.0  # do not penalize the bias
        w = np.zeros(d)
        self.n_iter_ = 0
        for it in range(self.max_iter):
            p = self._sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / n + reg @ w
            hess = (Xb.T * (p * (1 - p))) @ Xb / n + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            self.n_iter_ = it + 1
            if np.linalg.norm(step) < self.tol:
                break
        self.w_ = w
        return self

    def predict_proba(self, X):
        Xb = np.hstack([np.ones((len(X), 1)), self.scaler_.transform(np.asarray(X, float))])
        p1 = self._sigmoid(Xb @ self.w_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RandomForestModel:
    """Random Forest classifier using scikit-learn."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.model_.fit(np.asarray(X, float), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float))


class GradientBoostingModel:
    """Gradient Boosting classifier using scikit-learn."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05,
                 max_depth: int = 2, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingClassifier
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model_.fit(np.asarray(X, float), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return self.model_.predict(X)
