"""
Data loading and merging for the IBM Telco Customer Churn dataset.

Implements the merge pipeline from Section 9.1 of the paper. Handles the
six Excel files (main, population, demographics, location, services,
status), reconciles duplicate columns, and produces a single dataframe
ready for preprocessing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# File names expected in the data directory
# ---------------------------------------------------------------------------
FILE_MAIN = "Telco_customer_churn.xlsx"
FILE_POP = "Telco_customer_churn_population.xlsx"
FILE_DEM = "Telco_customer_churn_demographics.xlsx"
FILE_LOC = "Telco_customer_churn_location.xlsx"
FILE_SERV = "Telco_customer_churn_services.xlsx"
FILE_STAT = "Telco_customer_churn_status.xlsx"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_telco_files(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all six IBM Telco Excel files from `data_dir`.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the six required .xlsx files.

    Returns
    -------
    dict mapping short keys ('main', 'pop', 'dem', 'loc', 'serv', 'stat')
    to their respective DataFrames.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = {
        "main": FILE_MAIN,
        "pop": FILE_POP,
        "dem": FILE_DEM,
        "loc": FILE_LOC,
        "serv": FILE_SERV,
        "stat": FILE_STAT,
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Required file missing: {fpath}. See data/README.md."
            )
        frames[key] = pd.read_excel(fpath)
        print(f"  Loaded {fname:50s} shape={frames[key].shape}")
    return frames


def merge_telco_data(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the six Telco files into one cleaned DataFrame.

    Pipeline:
      1. Start with the main file.
      2. Attach Population from the population file via Zip Code.
      3. Normalize the Customer ID column across files.
      4. Merge demographics, services, and status files on Customer ID,
         dropping duplicate columns introduced by the merges.
      5. Drop identifier and geographic columns that should not be features.
      6. Drop any leftover `_x` / `_y` suffix columns from the merges.

    Parameters
    ----------
    frames : dict
        Output of `load_telco_files`.

    Returns
    -------
    pd.DataFrame
        Merged dataframe, ready for preprocessing.
    """
    df = frames["main"].copy()
    print(f"Initial shape: {df.shape}")

    df = _attach_population(df, frames["pop"])
    df = _normalize_customer_id(df)

    df_dem = _prepare_demographics(frames["dem"])
    df_serv = frames["serv"].copy()
    df_stat = frames["stat"].copy()

    df = _merge_services(df, df_serv)
    df = _merge_status(df, df_stat)
    df = _drop_non_feature_columns(df)
    df = _drop_xy_suffixes(df)

    print(f"Final merged shape: {df.shape}")
    return df


def load_and_merge(data_dir: str | Path) -> pd.DataFrame:
    """Convenience: load all 6 files and merge in one call."""
    return merge_telco_data(load_telco_files(data_dir))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _attach_population(df: pd.DataFrame, df_pop: pd.DataFrame) -> pd.DataFrame:
    """Join the Population column from df_pop using Zip Code as key."""
    df = df.copy()
    df_pop = df_pop.copy()
    df["Zip Code"] = df["Zip Code"].astype(str).str.zfill(5)
    df_pop["Zip Code"] = df_pop["Zip Code"].astype(str).str.zfill(5)
    df = pd.merge(df, df_pop[["Zip Code", "Population"]], on="Zip Code", how="left")
    print(f"  After population merge: {df.shape}")
    return df


def _normalize_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Some files use 'CustomerID', others 'Customer ID'. Standardize."""
    return df.rename(columns={"CustomerID": "Customer ID"})


def _prepare_demographics(df_dem: pd.DataFrame) -> pd.DataFrame:
    """Clean the demographics file: trim column names, drop duplicates."""
    df_dem = df_dem.rename(columns={"CustomerID": "Customer ID"})
    df_dem.columns = df_dem.columns.str.strip()
    if df_dem["Customer ID"].duplicated().any():
        df_dem = df_dem.drop_duplicates(subset=["Customer ID"], keep="first")
    return df_dem


def _merge_services(df: pd.DataFrame, df_serv: pd.DataFrame) -> pd.DataFrame:
    """Merge services file, then drop columns duplicated from the main file."""
    df = pd.merge(df, df_serv, on="Customer ID", how="left", suffixes=("", "_serv"))
    duplicates = [c for c in df.columns if c.endswith("_serv")]
    if duplicates:
        df = df.drop(columns=duplicates)
    print(f"  After services merge: {df.shape}")
    return df


def _merge_status(df: pd.DataFrame, df_stat: pd.DataFrame) -> pd.DataFrame:
    """Merge selected columns from the status file."""
    if df_stat["Customer ID"].duplicated().any():
        df_stat = df_stat.drop_duplicates(subset=["Customer ID"], keep="first")

    wanted = ["Customer ID", "Satisfaction Score", "Customer Status", "Churn Category"]
    available = [c for c in wanted if c in df_stat.columns]
    df = pd.merge(df, df_stat[available], on="Customer ID", how="left")
    print(f"  After status merge: {df.shape}")
    return df


# Columns that are identifiers or geographic detail — not features.
NON_FEATURE_COLUMNS = [
    "Customer ID", "Location ID", "Service ID", "Status ID",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
    "Country", "State", "City", "Count", "Quarter", "Population",
]


def _drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier and geographic columns."""
    to_drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  Dropped {len(to_drop)} non-feature columns, shape={df.shape}")
    return df


def _drop_xy_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leftover `_x` and `_y` suffix columns from earlier merges."""
    to_drop = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df
