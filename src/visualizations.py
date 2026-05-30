"""
Plotting helpers — reproduces every figure in the paper.

Each function returns a `matplotlib.Figure` so the caller can save it,
display it inline, or compose it with other plots. None of these
functions call `plt.show()` — that's up to the caller.

Figures map to the paper as follows:

    Figure 1: plot_accuracy_vs_disparity      (Section 9.2)
    Figure 2: plot_fairness_accuracy_tradeoff (Section 9.3)
    Figure 3: plot_segment_breakdown          (Section 9.4)
    Figure 4: plot_accuracy_by_gender         (Section 9.4)
    Figure 5: plot_roc_by_contract            (Section 9.5)
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


_DEFAULT_PALETTE = ["#7EC8E3", "#F08080", "#90EE90"]


# ---------------------------------------------------------------------------
# Figure 1 — Accuracy vs Segment Disparity bar chart
# ---------------------------------------------------------------------------
def plot_accuracy_vs_disparity(results_df: pd.DataFrame) -> plt.Figure:
    """Side-by-side bars: accuracy and segment disparity per model.

    Expects a DataFrame with columns: 'Model', 'accuracy', 'disparity'.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width / 2, results_df["accuracy"], width,
           label="Accuracy", color="#7EC8E3")
    ax.bar(x + width / 2, results_df["disparity"], width,
           label="Segment Disparity", color="#F08080")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy vs Segment Disparity")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Pareto frontier: fairness penalty vs accuracy
# ---------------------------------------------------------------------------
def plot_fairness_accuracy_tradeoff(fair_df: pd.DataFrame) -> plt.Figure:
    """Pareto frontier across λ_fairness values.

    Expects columns: 'lambda_f', 'accuracy', 'disparity'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fair_df["accuracy"], fair_df["disparity"],
            "o-", color="coral", linewidth=2, markersize=8)

    ax.set_xlabel("Global Accuracy", fontsize=12)
    ax.set_ylabel("Max Segment Mean Error Disparity", fontsize=12)
    ax.set_title("Fairness-Accuracy Trade-off (Contract Type)", fontsize=14)

    for _, row in fair_df.iterrows():
        ax.annotate(
            f"λ_f={row['lambda_f']}",
            (row["accuracy"], row["disparity"]),
            xytext=(5, 5), textcoords="offset points", fontsize=9,
        )

    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Per-segment accuracy + mean error
# ---------------------------------------------------------------------------
def plot_segment_breakdown(contract_metrics: pd.DataFrame) -> plt.Figure:
    """Two-panel: accuracy bar chart + mean error bar chart by Contract type.

    Expects columns: 'Contract', 'accuracy', 'mean_error'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: accuracy
    bars1 = ax1.bar(contract_metrics["Contract"],
                    contract_metrics["accuracy"],
                    color=_DEFAULT_PALETTE[: len(contract_metrics)])
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Contract Type")
    ax1.set_ylim(0.7, 1.0)
    for bar, val in zip(bars1, contract_metrics["accuracy"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom")

    # Right: mean error
    bars2 = ax2.bar(contract_metrics["Contract"],
                    contract_metrics["mean_error"],
                    color=_DEFAULT_PALETTE[: len(contract_metrics)])
    ax2.set_ylabel("Mean Prediction Error")
    ax2.set_title("Mean Error by Contract Type")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars2, contract_metrics["mean_error"]):
        offset = 0.002 if val >= 0 else -0.005
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + offset,
                 f"{val:.4f}", ha="center",
                 va="bottom" if val >= 0 else "top")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Accuracy by Gender
# ---------------------------------------------------------------------------
def plot_accuracy_by_gender(gender_metrics: pd.DataFrame) -> plt.Figure:
    """Single-panel bar chart of accuracy by Gender.

    Expects columns: 'Gender', 'accuracy'.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(gender_metrics["Gender"], gender_metrics["accuracy"],
                  color=["#ADD8E6", "#FFB6C1"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Gender")
    ax.set_ylim(0.8, 0.9)
    for bar, val in zip(bars, gender_metrics["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — ROC curves per Contract type
# ---------------------------------------------------------------------------
def plot_roc_by_contract(
    df_analysis: pd.DataFrame,
    proba_col: str = "fair_pred_proba",
    title: str = "ROC Curves by Contract - Fair Model (λ=2)",
) -> plt.Figure:
    """One ROC curve per Contract type with AUC in the legend.

    Expects df_analysis to have columns: 'Contract', 'y_true', and the
    probability column named in `proba_col`.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for contract in df_analysis["Contract"].unique():
        mask = df_analysis["Contract"] == contract
        y_true = df_analysis.loc[mask, "y_true"]
        y_score = df_analysis.loc[mask, proba_col]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{contract} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: save a figure to outputs/figures/
# ---------------------------------------------------------------------------
def savefig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save a figure with tight bounding box and given DPI."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
