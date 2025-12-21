"""features.py

Feature engineering helpers and small metric utilities used across Step 4.

Main idea
---------
We intentionally split feature engineering into two phases:

1) `compute_fe_params(TRAIN)`
   Learns any *data-dependent* thresholds/edges **only on TRAIN**.
   Examples: max distances for bucket edges, the 75th percentile for the amenity score.
   This avoids using information from TEST when defining transformations.

2) `_feature_engineering_for_ml(df, fe_params)`
   Applies the same transformations to TRAIN and TEST using the pre-computed parameters.

The functions here are used by regression, classification, and clustering so that all tasks
operate on a consistent, comparable feature space.

Notes
-----
- This module does not fit ML models; it only prepares features.
- Outlier filtering is provided as a generic helper (IQR rule) and must be applied
  deliberately by the caller (and consistently with the chosen evaluation scope).
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² safely.

    We compute per-city metrics in several places. Some cities may have very small
    sample sizes in TEST, which can make R² undefined or misleading.

    Returns
    -------
    float
        R² if at least 2 samples are available, otherwise NaN.
    """
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def compute_fe_params(df: pd.DataFrame) -> Dict[str, float]:
    """Fit feature-engineering parameters on TRAIN ONLY.

    These parameters define thresholds/bucket edges that would otherwise depend on the
    distribution of the data. Fitting them on TRAIN prevents information from TEST from
    influencing feature construction.

    Parameters
    ----------
    df:
        Training DataFrame.

    Returns
    -------
    Dict[str, float]
        Dictionary of fitted parameters used by `_feature_engineering_for_ml`.

    Keys
    ----
    max_center_dist:
        Upper edge for the `distance_bucket` bins (minimum enforced).
    max_metro_dist:
        Upper edge for the `metro_dist_bucket` bins (minimum enforced).
    amenity_q75:
        75th percentile of `amenity_score` used for `is_luxury`.
    """
    # Max distance to city center used as the top bucket edge (with a minimum safeguard).
    max_center_dist = pd.to_numeric(df.get("dist"), errors="coerce").max()
    if pd.isna(max_center_dist) or max_center_dist <= 6:
        max_center_dist = 6.0001

    # Max distance to metro used as the top bucket edge (with a minimum safeguard).
    max_metro_dist = pd.to_numeric(df.get("metro_dist"), errors="coerce").max()
    if pd.isna(max_metro_dist) or max_metro_dist <= 2:
        max_metro_dist = 2.0001

    # Amenity score and its 75th percentile (TRAIN) to define a simple "luxury" flag.
    attr = pd.to_numeric(df.get("attr_index_norm"), errors="coerce")
    rest = pd.to_numeric(df.get("rest_index_norm"), errors="coerce")
    amenity_score = (attr + rest) / 2
    amenity_q75 = amenity_score.quantile(0.75)
    if pd.isna(amenity_q75):
        amenity_q75 = 0.0

    return {
        "max_center_dist": float(max_center_dist),
        "max_metro_dist": float(max_metro_dist),
        "amenity_q75": float(amenity_q75),
    }


def remove_outliers_iqr(df: pd.DataFrame, col: str = "realSum", k: float = 1.5) -> pd.DataFrame:
    """Filter rows using the IQR rule on a single column.

    Parameters
    ----------
    df:
        Input DataFrame.
    col:
        Column on which the IQR rule is computed (default: `realSum`).
    k:
        IQR multiplier (default: 1.5). Larger values remove fewer points.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the DataFrame (index reset).

    Notes
    -----
    This helper does not decide *when* outlier filtering should be applied.
    The caller controls the evaluation scope, e.g.:
    - real-world evaluation: do not filter TEST
    - typical-case evaluation: apply TRAIN-derived bounds consistently
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame for outlier removal.")

    series = pd.to_numeric(df[col], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = series.between(lower, upper)
    return df.loc[mask].reset_index(drop=True)


def _feature_engineering_for_ml(df: pd.DataFrame, fe_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Apply the common feature engineering steps.

    Parameters
    ----------
    df:
        Input DataFrame (TRAIN or TEST). The function works on a copy.
    fe_params:
        Parameters returned by `compute_fe_params(TRAIN)`.
        If None, parameters are computed from `df` (useful for exploratory runs,
        but callers should pass TRAIN-fitted params during evaluation).

    Returns
    -------
    pd.DataFrame
        A transformed DataFrame with engineered features added and redundant
        raw columns removed.

    Expected input columns (minimum)
    -------------------------------
    bedrooms, person_capacity, metro_dist, dist,
    attr_index_norm, rest_index_norm,
    guest_satisfaction_overall, cleanliness_rating,
    host_is_superhost

    Features created
    ---------------
    beds_per_person:
        Bedroom density proxy (bedrooms / person_capacity).
    capacity_per_bedroom:
        Capacity normalized by bedrooms.
    capacity_gt2, is_studio:
        Simple binary flags.
    log_metro_dist, log_dist_center:
        Log-transformed distances.
    distance_ratio:
        Metro distance relative to center distance.
    distance_bucket, metro_dist_bucket, guest_satisfaction_bucket:
        Interpretable categorical buckets.
    amenity_score, is_luxury:
        Combined amenity signal and a premium flag.
    """
    if fe_params is None:
        fe_params = compute_fe_params(df)

    df = df.copy()

    # Capacity / layout features: simple ratios and flags to capture "density" and apartment type.
    # Note: `beds_per_person` uses `bedrooms` because the dataset does not provide a separate
    # "number of beds" column; it acts as a bedroom-density proxy.
    df["beds_per_person"] = (df["bedrooms"] / df["person_capacity"]).replace([np.inf, -np.inf], np.nan)
    df["capacity_per_bedroom"] = df["person_capacity"] / (df["bedrooms"] + 1)
    df["capacity_gt2"] = (df["person_capacity"] >= 3).astype(int)
    df["is_studio"] = (df["bedrooms"] == 0).astype(int)

    # Distance features: log transforms reduce skew; ratio captures metro-vs-center trade-off.
    df["log_metro_dist"] = np.log1p(df["metro_dist"])
    df["log_dist_center"] = np.log1p(df["dist"])
    df["distance_ratio"] = df["metro_dist"] / (df["dist"] + 1e-3)

    # Bucketize distance-to-center into coarse, interpretable categories.
    max_center_dist = fe_params.get("max_center_dist", None)
    if max_center_dist is None or pd.isna(max_center_dist) or max_center_dist <= 6:
        max_center_dist = 6.0001
    df["distance_bucket"] = pd.cut(
        df["dist"],
        bins=[-0.01, 2, 6, max_center_dist],
        labels=["center", "mid", "outer"],
        include_lowest=True,
    )

    # Amenity score and a simple "luxury" flag using a TRAIN-fitted threshold.
    df["amenity_score"] = (df["attr_index_norm"] + df["rest_index_norm"]) / 2

    amenity_q75 = fe_params.get("amenity_q75", None)
    if amenity_q75 is None or pd.isna(amenity_q75):
        amenity_q75 = df["amenity_score"].quantile(0.75)
    if pd.isna(amenity_q75):
        amenity_q75 = 0.0
    df["is_luxury"] = (
        (df["amenity_score"] >= amenity_q75)
        & (df["guest_satisfaction_overall"] >= 95)
        & (df["cleanliness_rating"] >= 9)
    ).astype(int)

    # Bucketize satisfaction into ordinal levels (helps both linear and tree models).
    df["guest_satisfaction_bucket"] = pd.cut(
        df["guest_satisfaction_overall"],
        bins=[0, 80, 90, 95, 100],
        labels=["low", "medium", "high", "excellent"],
        include_lowest=True,
    )

    # Normalize messy superhost values into a clean 0/1 indicator.
    super_raw = df["host_is_superhost"].astype(str).str.strip().str.lower()
    super_map = {"t": 1, "true": 1, "y": 1, "yes": 1, "1": 1}
    df["host_is_superhost"] = super_raw.map(super_map).fillna(0).astype(int)

    # Bucketize metro distance; uses TRAIN-fitted max edge to avoid TEST influence.
    max_dist = fe_params.get("max_metro_dist", None)
    if max_dist is None or pd.isna(max_dist) or max_dist <= 2:
        max_dist = 2.0001
    df["metro_dist_bucket"] = pd.cut(
        df["metro_dist"],
        bins=[-0.01, 0.5, 2, max_dist],
        labels=["near", "mid", "far"],
        include_lowest=True,
    )

    # Drop raw columns that are now represented by engineered features (avoid redundancy/collinearity).
    df = df.drop(columns=["metro_dist", "dist", "attr_index_norm", "rest_index_norm"])
    return df
