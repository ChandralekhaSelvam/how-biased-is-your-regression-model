"""
Feature preparation: missing-value handling, encoding, scaling, leak removal.

Implements the preprocessing pipeline from Sections 9.1 of the paper.
The leak-column removal is critical: several columns in the IBM Telco
dataset are computed *after* the churn event (Churn Score, CLTV,
Customer Status, Churn Category, Satisfaction Score) and would cause
data leakage if used as features.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Columns to drop because they leak the target or are derived from it
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Churn Value"

# Columns that directly encode the target or are computed from it.
TARGET_DERIVED = ["Churn Label", "Churn Score", "CLTV", "Churn Reason"]

# Columns that are post-hoc to churn and would leak.
LEAK_PATTERNS = ["Customer Status", "Churn Category", "CLTV", "Churn Score"]

# Additional explicit leak columns.
EXPLICIT_LEAKS = ["Satisfaction Score", "Total Revenue", "Total Refunds"]

# Categorical fill values used when missing
DEFAULT_FILLS = {
    "Churn Category": "Unknown",
    "Offer": "No Offer",
    "Internet Type": "None",
}

# Segment columns retained from the raw dataframe for fairness analysis
SEGMENT_COLUMNS = ["Contract", "Gender", "Senior Citizen"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def split_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and binary target (y).

    Removes target-derived columns from X and drops rows with missing target.
    """
    df = df[df[TARGET_COLUMN].notna()].copy()
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN] + TARGET_DERIVED, errors="ignore")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Target balance:\n{y.value_counts(normalize=True).round(3)}")
    return X, y


def fill_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Fill known missing-value patterns with sensible defaults."""
    X = X.copy()
    for col, fill_val in DEFAULT_FILLS.items():
        if col in X.columns:
            X[col] = X[col].fillna(fill_val)
    remaining = X.isnull().sum().sum()
    print(f"  Remaining missing values: {remaining}")
    return X


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode object columns. Drops first level to avoid collinearity."""
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print(f"  Encoded {len(cat_cols)} categorical cols. Shape: {X_encoded.shape}")
    return X_encoded


def scale_numericals(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale all numeric columns. Returns (scaled_X, fitted_scaler)."""
    X = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Boolean columns from get_dummies should be treated as numeric here.
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols].astype(float))
    return X, scaler


def remove_leak_columns(X_encoded: pd.DataFrame) -> pd.DataFrame:
    """Strip out all columns matching post-hoc leak patterns.

    Returns a new dataframe without:
      - any column matching LEAK_PATTERNS (e.g., 'Customer Status_Joined')
      - the EXPLICIT_LEAKS list
    """
    leak_cols = [
        c for c in X_encoded.columns
        if any(pattern in c for pattern in LEAK_PATTERNS)
    ]
    X_clean = X_encoded.drop(columns=leak_cols)
    X_clean = X_clean.drop(columns=EXPLICIT_LEAKS, errors="ignore")
    print(f"  Removed {len(leak_cols)} pattern-leak cols + explicit leaks. "
          f"Shape: {X_clean.shape}")
    return X_clean


def extract_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Pull out raw segment columns for fairness analysis.

    These are the *unencoded* segment columns (Contract, Gender,
    Senior Citizen) used for segment-level disparity calculation.
    """
    available = [c for c in SEGMENT_COLUMNS if c in df.columns]
    return df[available].copy()


# ---------------------------------------------------------------------------
# One-shot pipeline
# ---------------------------------------------------------------------------
def prepare(
    df: pd.DataFrame,
    drop_leaks: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """End-to-end feature prep: split → fill → encode → scale → remove leaks.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe from `data_loader.load_and_merge`.
    drop_leaks : bool
        Whether to strip leak columns (default True). Set False if you want
        to demonstrate the leakage effect.

    Returns
    -------
    X_clean : pd.DataFrame
        Final feature matrix, encoded and scaled.
    y : pd.Series
        Binary target.
    segments_df : pd.DataFrame
        Raw segment columns (Contract, Gender, Senior Citizen) aligned to X.
    """
    segments_df = extract_segments(df)
    X, y = split_features_and_target(df)
    X = fill_missing(X)
    X_encoded = encode_categoricals(X)
    X_scaled, _ = scale_numericals(X_encoded)
    X_clean = remove_leak_columns(X_scaled) if drop_leaks else X_scaled

    # Align segments to the cleaned features (in case any rows were dropped)
    segments_df = segments_df.loc[X_clean.index].copy()
    return X_clean, y, segments_df
