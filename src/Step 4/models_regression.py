"""Step 4 – Regression modelling & evaluation.

Goal
----
Train and evaluate multiple regression models to predict Airbnb booking revenue (`realSum`).
Regression is the *main task* of the project.

Target handling
---------------
We train on a log-transformed target to reduce skew:
- y_train = log1p(realSum)
During evaluation we convert predictions back to Euro scale:
- y_pred_eur = expm1(y_pred_log) and clip to >= 0
All reported metrics (RMSE / MAE / R²) are computed on the *Euro scale*.

Leakage prevention
------------------
- Splitting is handled by `make_train_test_split(...)` (typically city-stratified).
- Feature-engineering parameters are fitted on TRAIN only via `compute_fe_params(...)`
  and then applied unchanged to TEST.

Outlier sensitivity
-------------------
This module supports an optional IQR-based “typical-case” scope controlled by `drop_outliers`.
- IQR bounds are computed on TRAIN (after the split).
- TRAIN is filtered using those bounds.
- The same TRAIN-derived bounds are also applied to TEST so both sets represent the same scope.
  (This is intentional for the typical-case evaluation; the unfiltered run still reports full-scope results.)

What gets written to disk
-------------------------
Files are written into the provided `plots_dir` and `models_dir`:
- Baseline (city-mean) metrics and plots.
- Per-model global metrics table (`regression_model_comparison.csv`) + comparison plot.
- Per-city metrics (CSV + MAE plot) and RMSE-by-revenue-bucket plot for baseline and each model.
- Diagnostics per model: predicted-vs-actual and residual histogram.
- Optional feature importance plots for tree-based models.
- Serialized model pipelines (`.pkl`) for reproducibility.

Interpretation warning
----------------------
This is predictive modelling. Any association between features (e.g., crime indices) and revenue
reflects patterns learned by the model and must not be presented as causal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor

from config import RANDOM_STATE
from data_io import make_train_test_split
from features import compute_fe_params, _feature_engineering_for_ml, safe_r2


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing transformer used inside each regression pipeline.

    - Numeric features: median imputation + standard scaling.
    - Categorical features: most-frequent imputation + one-hot encoding.

    We keep preprocessing inside a single `ColumnTransformer` so that every model is trained and
    evaluated with the exact same feature preparation.
    """
    # Define which columns are treated as categorical vs numeric for preprocessing.
    categorical_features = [
        "room_type",
        "City",
        "metro_dist_bucket",
        "distance_bucket",
        "guest_satisfaction_bucket",
    ]
    # Everything not listed as categorical is treated as numeric.
    numeric_features = [c for c in feature_df.columns if c not in categorical_features]

    # Numeric pipeline: impute missing values and scale for models sensitive to feature magnitude.
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: impute and one-hot encode; unknown categories in TEST are ignored safely.
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine numeric + categorical preprocessing into a single transformer used inside each model pipeline.
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return expanded feature names after preprocessing."""
    return preprocessor.get_feature_names_out().tolist()


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics on the raw Euro scale.

    Metrics:
    - RMSE: penalizes large errors (sensitive to outliers)
    - MAE: average absolute error (more robust)
    - R²: explained variance (may be unstable for small groups; see `safe_r2` for per-city)
    """
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_pred_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_path: Path
) -> None:
    """Save a scatter plot of predicted vs. actual revenue (Euro scale).

    Points are colored red for overprediction and green for underprediction.
    The diagonal line represents perfect predictions.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # Color points by over/underprediction for visual diagnostic
    colors = ["red" if yp > yt else "green" for yt, yp in zip(y_true, y_pred)]
    ax.scatter(y_true, y_pred, alpha=0.5, s=12, c=colors)
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markersize=6, label="Overprediction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="green", markersize=6, label="Underprediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Error type")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="black", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual revenue (€)")
    ax.set_ylabel("Predicted revenue (€)")
    ax.set_title(f"Predicted vs Actual (raw scale) – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_path: Path
) -> None:
    """Save a residual histogram (actual − predicted) on Euro scale.

    This diagnostic is useful to spot systematic bias (shift away from 0) and
    heavy tails/outliers (very wide distribution).
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40, alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title(f"Residuals (raw scale) – {model_name}")
    ax.set_xlabel("Actual - Predicted (Euro)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, model_name: str, out_path: Path):
    """Save a horizontal bar plot for top-N feature importances.

    Important: Importances are model-specific and not causal. For tree/boosting
    models they indicate how strongly features were used to split the data.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("importance")
    ax.barh(df_sorted["feature"], df_sorted["importance"])
    ax.set_title(f"Top {len(df)} Features – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def feature_importance_df(pipeline: Pipeline, top_n: int = 10):
    """Extract a top-N feature-importance table from a fitted pipeline.

    Returns None if the underlying regressor does not expose `feature_importances_`.
    Feature names come from the fitted `ColumnTransformer`.
    """
    reg = pipeline.named_steps["regressor"]
    if not hasattr(reg, "feature_importances_"):
        return None
    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
    importances = pd.Series(reg.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    return top.reset_index().rename(columns={"index": "feature", 0: "importance"})


def train_for_split(
    base_df: pd.DataFrame,
    split_strategy: str,
    drop_outliers: bool,
    plots_dir: Path,
    models_dir: Path,
):
    """Run one full regression experiment for a given split and outlier setting.

    Parameters
    ----------
    base_df:
        Input dataframe containing the target column `realSum` and all features.
    split_strategy:
        Key understood by `make_train_test_split` (e.g., city-stratified 80/20).
    drop_outliers:
        If True, applies IQR outlier filtering on the TRAIN set only (no test leakage).
        The test set remains unchanged.
    plots_dir / models_dir:
        Output folders for plots/CSVs and serialized model artifacts.

    Side effects (files written)
    ---------------------------
    - `regression_model_comparison.csv` + `model_comparison_overall.png`
    - Baseline: `baseline_city_scores.csv`, `mae_by_city_city-mean_baseline.png`,
      `rmse_by_bucket_city-mean_baseline.png`
    - Per model: `city_scores_<model>.csv`, `mae_by_city_<model>.png`,
      `rmse_by_bucket_<model>.png`, `pred_vs_actual_<model>.png`, `residuals_<model>.png`,
      and (if supported) `feature_importance_<model>.png`
    - Best-model extras: `crime_ablation_metrics.csv` and optionally `pdp_crime_index.png`
    - Heatmap: `r2_by_city_and_model.csv` and `r2_heatmap_city_model.png`

    Returns
    -------
    pandas.DataFrame
        Model comparison table (sorted by RMSE).
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        LGBMRegressor = None

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        CatBoostRegressor = None

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None

    # Work on a copy so the caller's dataframe is never mutated.
    df = base_df.copy()
    # NOTE: Outlier handling is applied on the TRAIN set only (after the split)
    # to avoid leaking information from the test set.

    # Separate target (continuous revenue) from raw feature columns.
    y_raw_full = df["realSum"]
    # Train on log1p(revenue) to reduce skew; metrics are later computed on Euro scale.
    y_log_full = np.log1p(y_raw_full)
    X_raw_full = df.drop(columns=["realSum"])

    # Split into TRAIN/TEST according to the chosen strategy (often city-stratified).
    print(f"\n=== Running split: {split_strategy} | Outlier removal: {'ON' if drop_outliers else 'OFF'} ===")
    # Split is computed before any filtering/feature fitting to avoid leakage.
    X_train_raw, X_test_raw, y_train_log, y_test_log = make_train_test_split(
        X_raw_full, y_log_full, X_raw_full["City"], strategy_key=split_strategy
    )

    # Optional typical-case scope: compute IQR bounds on TRAIN and filter TRAIN; apply the same bounds to TEST.
    if drop_outliers:
        y_train_raw_tmp = pd.Series(np.expm1(y_train_log))
        q1 = y_train_raw_tmp.quantile(0.25)
        q3 = y_train_raw_tmp.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = y_train_raw_tmp.between(lower, upper)
        removed = int((~mask).sum())
        print(
            f"Train-only outlier removal on 'realSum': IQR={iqr:.2f}, lower={lower:.2f}, upper={upper:.2f}. "
            f"Removed {removed} of {len(mask)} train rows ({removed / len(mask) * 100:.2f}%)."
        )
        X_train_raw = X_train_raw.loc[mask.values].reset_index(drop=True)
        y_train_log = pd.Series(y_train_log).loc[mask.values].reset_index(drop=True)

        # Apply the same train-derived bounds to TEST to drop extreme values there as well.
        y_test_raw_tmp = pd.Series(np.expm1(y_test_log))
        mask_test = y_test_raw_tmp.between(lower, upper)
        removed_test = int((~mask_test).sum())
        X_test_raw = X_test_raw.loc[mask_test.values].reset_index(drop=True)
        y_test_log = pd.Series(y_test_log).loc[mask_test.values].reset_index(drop=True)
        print(
            f"Applied train IQR bounds to TEST: removed {removed_test} of {len(mask_test)} test rows "
            f"({removed_test / len(mask_test) * 100:.2f}%)."
        )

    # Fit feature-engineering parameters on TRAIN only, then apply the same transformation to TEST.
    fe_params = compute_fe_params(X_train_raw)
    X_train = _feature_engineering_for_ml(X_train_raw, fe_params=fe_params)
    X_test = _feature_engineering_for_ml(X_test_raw, fe_params=fe_params)

    # Convert log-target back to Euro scale for reporting (y = expm1(log1p(y))).
    y_train_raw = np.expm1(y_train_log)
    y_test_raw = np.expm1(y_test_log)

    # Build preprocessing on the engineered TRAIN feature set (consistent across all models).
    preprocessor = build_preprocessor(X_train)

    print(f"Loaded {len(df):,} rows.")
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    if drop_outliers:
        print("(Train size reflects train-only outlier filtering)")

    # Define the set of regression models to compare (linear + tree + ensemble + optional boosters).
    model_factories = {
        "Linear Regression (OLS)": lambda: LinearRegression(),
        "Ridge Regression": lambda: Ridge(alpha=1.0),
        "Decision Tree (depth=6)": lambda: DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Decision Tree (depth=12)": lambda: DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest (200 trees)": lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest (500 trees)": lambda: RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    if XGBRegressor is not None:
        model_factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
        )

    if LGBMRegressor is not None:
        model_factories["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            objective="regression",
            n_jobs=-1,
        )

    if CatBoostRegressor is not None:
        model_factories["CatBoost"] = lambda: CatBoostRegressor(
            n_estimators=400,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            verbose=False,
            random_seed=RANDOM_STATE,
        )

    # Baseline: predict using the mean TRAIN revenue per city (fallback to global mean).
    baseline_name = "City-mean Baseline"
    global_mean = y_train_raw.mean()
    city_means = y_train_raw.groupby(X_train["City"]).mean()
    baseline_preds = X_test["City"].map(city_means).fillna(global_mean).to_numpy()
    baseline_metrics = eval_metrics(y_test_raw, baseline_preds)

    baseline_test_df = X_test.copy()
    baseline_test_df["y_true"] = y_test_raw
    baseline_test_df["y_pred"] = baseline_preds

    baseline_city_scores = (
        baseline_test_df.groupby("City")
        .apply(
            lambda g: pd.Series({
                "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
                "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                "R2": safe_r2(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
            }),
            include_groups=False,
        )
    )
    print(f"\nPer-City Performance – {baseline_name}")
    print(baseline_city_scores.sort_values("MAE"))
    baseline_city_csv = plots_dir / "baseline_city_scores.csv"
    baseline_city_scores.reset_index().to_csv(baseline_city_csv, index=False)
    print(f"Saved baseline city scores → {baseline_city_csv}")

    fig, ax = plt.subplots(figsize=(7, 4))
    baseline_city_scores["MAE"].sort_values().plot(kind="bar", ax=ax)
    ax.set_title(f"MAE by City – {baseline_name}")
    ax.set_ylabel("MAE (Euro)")
    ax.set_xlabel("City")
    plt.tight_layout()
    baseline_city_mae_path = plots_dir / f"mae_by_city_{baseline_name.replace(' ', '_').lower()}.png"
    fig.savefig(baseline_city_mae_path, dpi=300)
    plt.close(fig)
    print(f"Saved MAE-by-city plot → {baseline_city_mae_path}")

    price_bins = [0, 50, 100, 200, 500, np.inf]
    price_labels = ["0–50", "50–100", "100–200", "200–500", "500+"]
    baseline_test_df["price_bucket"] = pd.cut(
        baseline_test_df["y_true"], bins=price_bins, labels=price_labels, include_lowest=True
    )
    bucket_rmse = (
        baseline_test_df.groupby("price_bucket", observed=True)
        .apply(lambda g: np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])), include_groups=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    bucket_rmse.plot(kind="bar", ax=ax)
    ax.set_title(f"RMSE by price bucket – {baseline_name}")
    ax.set_ylabel("RMSE (Euro)")
    ax.set_xlabel("Price bucket")
    plt.tight_layout()
    bucket_path = plots_dir / f"rmse_by_bucket_{baseline_name.replace(' ', '_').lower()}.png"
    fig.savefig(bucket_path, dpi=300)
    plt.close(fig)
    print(f"Saved RMSE-by-bucket plot → {bucket_path}")

    # Include the city-mean baseline in the comparison table/plot (train metrics not applicable).
    results = [
        {
            "Model": baseline_name,
            **baseline_metrics,
            "Train_RMSE": np.nan,
            "Train_MAE": np.nan,
            "Train_R2": np.nan,
        }
    ]
    all_city_scores: Dict[str, pd.DataFrame] = {}

    # Train and evaluate each model using the same preprocessing + evaluation protocol.
    for model_name, factory in model_factories.items():
        print(f"\n--- Training {model_name} ---")
        reg = factory()

        # Full pipeline: preprocessing -> regressor (prevents train/test preprocessing mismatch).
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", reg),
        ])

        # Fit on TRAIN (log-scale target).
        pipe.fit(X_train, y_train_log)

        # Predict on TEST, then convert predictions back to Euro scale for metrics.
        preds_log = pipe.predict(X_test)
        preds_raw = np.expm1(preds_log)
        preds_raw = np.clip(preds_raw, 0, None)
        metrics = eval_metrics(y_test_raw, preds_raw)

        # Also compute TRAIN metrics as an overfitting sanity check (not used for model selection).
        train_preds_log = pipe.predict(X_train)
        train_preds_raw = np.expm1(train_preds_log)
        train_preds_raw = np.clip(train_preds_raw, 0, None)
        train_metrics = eval_metrics(y_train_raw, train_preds_raw)
        print(
            f"Train metrics – {model_name}: RMSE={train_metrics['RMSE']:.2f}, MAE={train_metrics['MAE']:.2f}, R2={train_metrics['R2']:.3f}"
        )

        results.append({
            "Model": model_name,
            **metrics,
            "Train_RMSE": train_metrics["RMSE"],
            "Train_MAE": train_metrics["MAE"],
            "Train_R2": train_metrics["R2"],
        })

        # Persist the fitted pipeline so results can be reproduced without retraining.
        model_path = models_dir / f"model_{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved model → {model_path}")

        plot_pred_vs_actual(y_test_raw, preds_raw, model_name,
                            plots_dir / f"pred_vs_actual_{model_name.replace(' ', '_').lower()}.png")
        plot_residuals(y_test_raw, preds_raw, model_name,
                       plots_dir / f"residuals_{model_name.replace(' ', '_').lower()}.png")

        fi = feature_importance_df(pipe, top_n=10)
        if fi is not None:
            fi_path = plots_dir / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            plot_feature_importance(fi, model_name, fi_path)
            print(f"Top 10 Features for {model_name}:\n{fi}")

        test_df = X_test.copy()
        test_df["y_true"] = y_test_raw
        test_df["y_pred"] = preds_raw

        # Per-city metrics: check whether the model generalises consistently across cities.
        city_scores = (
            test_df.groupby("City")
            .apply(
                lambda g: pd.Series({
                    "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
                    "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                    "R2": safe_r2(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                }),
                include_groups=False,
            )
        )

        print(f"\nPer-City Performance – {model_name}")
        print(city_scores.sort_values("MAE"))

        all_city_scores[model_name] = city_scores
        city_scores_csv = plots_dir / f"city_scores_{model_name.replace(' ', '_').lower()}.csv"
        city_scores.reset_index().to_csv(city_scores_csv, index=False)
        print(f"Saved city scores → {city_scores_csv}")

        fig, ax = plt.subplots(figsize=(7, 4))
        city_scores["MAE"].sort_values().plot(kind="bar", ax=ax)
        ax.set_title(f"MAE by City – {model_name}")
        ax.set_ylabel("MAE (Euro)")
        ax.set_xlabel("City")
        plt.tight_layout()
        city_mae_path = plots_dir / f"mae_by_city_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(city_mae_path, dpi=300)
        plt.close(fig)
        print(f"Saved MAE-by-city plot → {city_mae_path}")

        test_df["price_bucket"] = pd.cut(test_df["y_true"], bins=price_bins, labels=price_labels, include_lowest=True)
        bucket_rmse = (
            test_df.groupby("price_bucket", observed=True)
            .apply(lambda g: np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])), include_groups=False)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        bucket_rmse.plot(kind="bar", ax=ax)
        ax.set_title(f"RMSE by price bucket – {model_name}")
        ax.set_ylabel("RMSE (Euro)")
        ax.set_xlabel("Price bucket")
        plt.tight_layout()
        bucket_path = plots_dir / f"rmse_by_bucket_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(bucket_path, dpi=300)
        plt.close(fig)
        print(f"Saved RMSE-by-bucket plot → {bucket_path}")

        if model_name.startswith("Decision Tree"):
            fig, ax = plt.subplots(figsize=(20, 10))
            feature_names = get_feature_names(pipe.named_steps["preprocessor"])
            plot_tree(
                pipe.named_steps["regressor"],
                feature_names=feature_names,
                filled=True,
                max_depth=3,
                ax=ax,
            )
            safe_name = (
                model_name.replace(" ", "_")
                .replace("=", "")
                .replace("(", "")
                .replace(")", "")
                .lower()
            )
            tree_path = plots_dir / f"decision_tree_structure_{safe_name}.png"
            fig.savefig(tree_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved decision tree visualization → {tree_path}")

    # Collect all models (including baseline) into one comparison table (sorted by RMSE).
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n=== FINAL MODEL PERFORMANCE (RAW SCALE) ===")
    print(results_df)
    results_csv_path = plots_dir / "regression_model_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved regression comparison table → {results_csv_path}")

    # Create a compact visual comparison across models for the report appendix.
    # Horizontal comparison plot to keep long model names readable.
    display_name = {
        "City-mean Baseline": "Baseline",
        "Random Forest (200 trees)": "RF 200",
        "Random Forest (500 trees)": "RF 500",
        "Decision Tree (depth=6)": "DT d=6",
        "Decision Tree (depth=12)": "DT d=12",
        "Linear Regression (OLS)": "LinReg",
        "Ridge Regression": "Ridge",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "CatBoost": "CatBoost",
    }
    results_df["Model_short"] = results_df["Model"].map(display_name).fillna(results_df["Model"])

    # Sort once by RMSE and reuse ordering for all panels
    sorted_df = results_df.sort_values("RMSE")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    sorted_df.plot(x="Model_short", y="RMSE", kind="barh", ax=axes[0], legend=False)
    axes[0].set_title("RMSE by Model")
    axes[0].set_xlabel("RMSE")
    axes[0].invert_yaxis()

    sorted_df.plot(x="Model_short", y="MAE", kind="barh", ax=axes[1], legend=False)
    axes[1].set_title("MAE by Model")
    axes[1].set_xlabel("MAE")
    axes[1].invert_yaxis()

    sorted_df.plot(x="Model_short", y="R2", kind="barh", ax=axes[2], legend=False)
    axes[2].set_title("$R^2$ by Model")
    axes[2].set_xlabel("$R^2$")
    axes[2].invert_yaxis()

    plt.tight_layout()
    overall_path = plots_dir / "model_comparison_overall.png"
    fig.savefig(overall_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall model comparison plot → {overall_path}")

    # Optional: build an R² heatmap across cities and models to visualise cross-city robustness.
    if all_city_scores:
        r2_frames = []
        for model_name, scores in all_city_scores.items():
            if "R2" in scores.columns:
                s = scores["R2"].rename(model_name)
                r2_frames.append(s)

        if r2_frames:
            r2_df = pd.concat(r2_frames, axis=1)
            r2_table_path = plots_dir / "r2_by_city_and_model.csv"
            r2_df.reset_index().to_csv(r2_table_path, index=False)
            print(f"Saved R^2 by city/model table → {r2_table_path}")

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(r2_df.values, aspect="auto")
            ax.set_xticks(range(len(r2_df.columns)))
            ax.set_xticklabels(r2_df.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(r2_df.index)))
            ax.set_yticklabels(r2_df.index)
            ax.set_title("$R^2$ by City and Model")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("$R^2$")
            plt.tight_layout()
            heatmap_path = plots_dir / "r2_heatmap_city_model.png"
            fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved R^2 heatmap by city and model → {heatmap_path}")

    return results_df
