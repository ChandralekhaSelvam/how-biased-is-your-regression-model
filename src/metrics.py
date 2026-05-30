"""
Fairness and segment-level metrics for the paper.

Implements the metric definitions from Sections 5.2 and 8.3:

    DP_diff           — Demographic Parity Difference (Section 5.2.1)
    EO_diff           — Equalized Odds Difference     (Section 5.2.2)
    MeanError_s       — Mean residual per segment     (Section 5.2.3)
    D_segment_disparity — Maximum pairwise disparity  (Section 8.3)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ArrayLike = Sequence | np.ndarray | pd.Series


# ---------------------------------------------------------------------------
# Core segment-level metrics
# ---------------------------------------------------------------------------
def mean_error_by_segment(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    segments: ArrayLike,
) -> dict:
    """Mean residual (y_true − y_pred_proba) for each segment.

    A positive value means the model *underpredicts* the positive class
    for that segment — exactly the pattern that leads to missed retention
    opportunities in churn modeling.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    segments = np.asarray(segments)

    result = {}
    for seg in np.unique(segments):
        mask = segments == seg
        if mask.sum() > 0:
            result[seg] = float(np.mean(y_true[mask] - y_pred_proba[mask]))
    return result


def segment_disparity(
    model,
    X,
    y: ArrayLike,
    segments: ArrayLike,
) -> float:
    """Maximum-to-minimum segment mean-error gap (D_segment_disparity proxy).

    Fits the model on (X, y), predicts probabilities, then returns
    max(MeanError) − min(MeanError) across segments. This is the
    operational version of the formula in Section 8.3.

    Returns
    -------
    float
        Range of segment mean errors. 0 = perfectly fair, larger = more biased.
    """
    model.fit(X, y)
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X)[:, 1]
    else:
        y_pred = model.predict(X)

    errors = mean_error_by_segment(y, y_pred, segments)
    return max(errors.values()) - min(errors.values())


def d_segment_disparity_squared(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    segments: ArrayLike,
) -> float:
    """Closed-form D_segment_disparity from Section 8.3 (squared form):

        D = max_{s, s'} (MeanError_s − MeanError_s')²
    """
    errors = mean_error_by_segment(y_true, y_pred_proba, segments)
    vals = list(errors.values())
    if len(vals) < 2:
        return 0.0
    return float((max(vals) - min(vals)) ** 2)


# ---------------------------------------------------------------------------
# Demographic Parity and Equalized Odds
# ---------------------------------------------------------------------------
def demographic_parity_difference(
    y_pred: ArrayLike,
    segments: ArrayLike,
    privileged_label=None,
    unprivileged_label=None,
) -> float:
    """DP_diff = P(ŷ=1 | s=unpriv) − P(ŷ=1 | s=priv).

    If `privileged_label` / `unprivileged_label` are not supplied, returns
    the maximum minus the minimum positive-prediction rate across segments.
    """
    y_pred = np.asarray(y_pred)
    segments = np.asarray(segments)

    rates = {}
    for seg in np.unique(segments):
        mask = segments == seg
        if mask.sum() > 0:
            rates[seg] = float(np.mean(y_pred[mask]))

    if privileged_label is not None and unprivileged_label is not None:
        return rates[unprivileged_label] - rates[privileged_label]
    return max(rates.values()) - min(rates.values())


def equalized_odds_difference(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    segments: ArrayLike,
) -> float:
    """EO_diff = max(|ΔFPR|, |ΔTPR|) across segments (Section 5.2.2)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    segments = np.asarray(segments)

    tpr, fpr = {}, {}
    for seg in np.unique(segments):
        mask = segments == seg
        yt = y_true[mask]
        yp = y_pred[mask]

        positives = yt == 1
        negatives = yt == 0
        tpr[seg] = float(yp[positives].mean()) if positives.any() else np.nan
        fpr[seg] = float(yp[negatives].mean()) if negatives.any() else np.nan

    tpr_range = max(tpr.values()) - min(tpr.values())
    fpr_range = max(fpr.values()) - min(fpr.values())
    return max(abs(tpr_range), abs(fpr_range))


# ---------------------------------------------------------------------------
# Convenience: per-segment metric table
# ---------------------------------------------------------------------------
def segment_metrics_table(
    df: pd.DataFrame,
    segment_col: str,
    y_true_col: str = "y_true",
    y_pred_class_col: str = "y_pred_class",
    residual_col: str = "residual",
) -> pd.DataFrame:
    """Per-segment accuracy / F1 / mean-error breakdown.

    Returns a tidy DataFrame with one row per segment value, suitable for
    feeding directly into the plotting helpers.
    """
    rows = []
    for val in df[segment_col].unique():
        subset = df[df[segment_col] == val]
        rows.append({
            segment_col: val,
            "count": len(subset),
            "accuracy": float((subset[y_pred_class_col] == subset[y_true_col]).mean()),
            "f1": float(f1_score(subset[y_true_col], subset[y_pred_class_col],
                                 zero_division=0)),
            "mean_error": float(subset[residual_col].mean()),
        })
    return pd.DataFrame(rows)
