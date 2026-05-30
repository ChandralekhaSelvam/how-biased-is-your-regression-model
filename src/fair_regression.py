"""
Fair Logistic Regression with a tunable segment disparity penalty.

Implements the fairness-penalized objective from Section 8 of the paper:

    L_total = L_prediction + λ_complexity · ||β||²
                            + λ_fairness   · D_segment_disparity

where D_segment_disparity is the variance of segment-conditional mean
errors. The optimizer uses scipy's BFGS to minimize the combined loss.

This is a scikit-learn compatible estimator: implements `fit`,
`predict`, and `predict_proba`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


class FairLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression with optional segment-level fairness penalty.

    Parameters
    ----------
    lambda_complexity : float, default=0.1
        L2 regularization strength on coefficients (the standard ridge term).
    lambda_fairness : float, default=0.0
        Weight on the segment disparity penalty. When 0, this reduces to
        plain L2-regularized logistic regression. Higher values trade
        predictive accuracy for segment equity.
    segments : array-like of shape (n_samples,), optional
        Segment labels (e.g., Contract type) used to compute the
        fairness penalty. Required when `lambda_fairness > 0`.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Fitted weight vector.
    intercept_ : float
        Fitted bias term.
    """

    def __init__(
        self,
        lambda_complexity: float = 0.1,
        lambda_fairness: float = 0.0,
        segments: Sequence | None = None,
    ):
        self.lambda_complexity = lambda_complexity
        self.lambda_fairness = lambda_fairness
        self.segments = segments

    # ------------------------------------------------------------------
    # Internal: sigmoid and loss
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid (clipped to avoid exp overflow)."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        segment_labels: np.ndarray | None,
    ) -> float:
        """Combined loss: log loss + L2 ridge + segment disparity penalty."""
        coef = beta[:-1]
        intercept = beta[-1]
        z = X @ coef + intercept
        y_pred = self._sigmoid(z)

        eps = 1e-15
        logloss = -np.mean(
            y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
        )

        complexity = self.lambda_complexity * 0.5 * np.sum(coef ** 2)

        fairness = 0.0
        if self.lambda_fairness > 0 and segment_labels is not None:
            unique_segments = np.unique(segment_labels)
            seg_errors = []
            for seg in unique_segments:
                mask = segment_labels == seg
                if mask.sum() > 0:
                    err = np.mean(y[mask] - y_pred[mask])
                    seg_errors.append(err)
            if len(seg_errors) > 1:
                # Variance of mean errors approximates D_segment_disparity
                # as a smooth, differentiable surrogate for the max-pair form.
                fairness = self.lambda_fairness * np.var(seg_errors)

        return logloss + complexity + fairness

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the model. `X` and `y` are coerced to float64 numpy arrays."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_features = X.shape[1]
        beta0 = np.zeros(n_features + 1, dtype=np.float64)

        segment_labels = (
            np.asarray(self.segments) if self.segments is not None else None
        )

        result = minimize(
            self._loss,
            beta0,
            args=(X, y, segment_labels),
            method="BFGS",
        )
        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]
        self.optimization_result_ = result
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return (n_samples, 2) array of [P(y=0), P(y=1)]."""
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.coef_ + self.intercept_
        p1 = self._sigmoid(z)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return hard class predictions at the given probability threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
