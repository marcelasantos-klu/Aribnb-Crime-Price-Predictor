"""
Train Airbnb price models with optional outlier removal and configurable splits.

This module combines the functionality of the previous two scripts:
- Prompt (or CLI flag) to choose whether outliers (IQR on target) are removed.
- Prompt (or CLI flag) to choose the train/test split strategy.
Outputs go to the corresponding subfolder under plots&models/WithOutliers or
plots&models/WithoutOutliers.
"""

from __future__ import annotations

import argparse
import ctypes
import os
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor


# ===============================================================
# PATHS & CONFIGURATION
# ===============================================================

DATA_PATH = Path("data/FinalFile/FinalDataSet_geo_merged.csv")
RANDOM_STATE = 42
OUTLIER_FOLDERS = {
    False: {"plots": Path("plots&models/WithOutliers/plots"), "models": Path("plots&models/WithOutliers/models")},
    True: {"plots": Path("plots&models/WithoutOutliers/plots"), "models": Path("plots&models/WithoutOutliers/models")},
}
# Add more split options here; switch via prompt or --split.
SPLIT_STRATEGIES = {
    "city_stratified_80_20_seed42": {"test_size": 0.2, "random_state": 42, "stratify_by_city": True},
    "city_stratified_70_30_seed42": {"test_size": 0.3, "random_state": 42, "stratify_by_city": True},
    "city_stratified_80_20_seed99": {"test_size": 0.2, "random_state": 99, "stratify_by_city": True},
}
DEFAULT_SPLIT_STRATEGY = "city_stratified_80_20_seed42"


# ===============================================================
# OPTIONAL LIBOMP FOR LIGHTGBM ON MACOS
# ===============================================================

def load_libomp_if_available() -> None:
    """Try to load the libomp runtime on macOS to avoid LightGBM/XGBoost errors.

    XGBoost and LightGBM depend on OpenMP (libomp) for multi-threading. On macOS
    this library is sometimes not on the default search path, which can lead to
    runtime import errors. This helper attempts to load libomp from a few common
    installation locations and sets the corresponding environment variables so
    the native libraries of the gradient-boosting frameworks can be initialized
    correctly. If nothing can be loaded, the script still runs, but some models
    may fail at import or fall back to single-threaded execution.
    """
    candidates = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
        Path.home() / "Library" / "Python" / "3.9" / "lib" / "python" / "site-packages" / "torch" / "lib" / "libomp.dylib",
    ]
    for path in candidates:
        if path.exists():
            try:
                ctypes.cdll.LoadLibrary(str(path))
                os.environ.setdefault("DYLD_LIBRARY_PATH", str(path.parent))
                os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(path.parent))
                print(f"Loaded libomp from: {path}")
                return
            except OSError:
                pass
    print("Warning: libomp not found. LightGBM may fail on macOS.")


# ===============================================================
# DATA LOADING
# ===============================================================

def load_data() -> pd.DataFrame:
    """Load the premerged Airbnb dataset and apply basic column cleanup.

    The function reads the FinalDataSet_geo_merged.csv file, normalizes the
    crime column name to `Crime_Index` (so the rest of the code can rely on a
    consistent identifier) and drops the synthetic `geo_id` column that was
    only used for grouping. All further preprocessing and feature engineering
    builds on the DataFrame returned here.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Crime Index": "Crime_Index"})
    df = df.drop(columns=["geo_id"], errors="ignore")
    return df


def prompt_split_strategy(default: str = DEFAULT_SPLIT_STRATEGY) -> str:
    """Interactively ask the user which train/test split configuration to use.

    The available options are taken from the global SPLIT_STRATEGIES dictionary.
    Each option specifies the test size, the random seed and whether the split
    is stratified by city. This prompt is only used when no `--split` argument
    is provided on the command line.

    The user can either:
    - enter a number (1..N) corresponding to one of the printed strategies,
    - type the strategy name directly, or
    - press ENTER to accept the default value.

    Invalid input falls back to the given default so that the script always has
    a valid and reproducible split configuration.

    Parameters
    ----------
    default : str
        Key of the strategy to use when the user presses ENTER or inputs an
        invalid value.

    Returns
    -------
    str
        The chosen split strategy key, guaranteed to exist in SPLIT_STRATEGIES.
    """
    options = list(SPLIT_STRATEGIES.keys())
    print("Select split strategy:")
    for idx, key in enumerate(options, 1):
        print(f"[{idx}] {key}")

    prompt = f"Enter choice 1-{len(options)} or name (default {default}): "
    try:
        user_input = input(prompt).strip()
    except EOFError:
        user_input = ""

    if not user_input:
        return default
    if user_input.isdigit():
        choice = int(user_input)
        if 1 <= choice <= len(options):
            return options[choice - 1]
    if user_input in SPLIT_STRATEGIES:
        return user_input

    print(f"Invalid input '{user_input}', using default {default}")
    return default


def prompt_outlier_choice(default_drop: bool = False) -> bool:
    """Ask the user whether to remove target outliers before training.

    This prompt is used when no explicit `--drop-outliers` flag is passed on the
    command line. It controls whether the IQR-based filter on the target column
    `realSum` is applied. Removing extreme price outliers makes the models more
    stable and leads to a more meaningful evaluation of typical listings.

    Parameters
    ----------
    default_drop : bool
        If True, pressing ENTER will enable outlier removal; if False, pressing
        ENTER will keep outliers.

    Returns
    -------
    bool
        True if outliers should be removed, False otherwise.
    """
    prompt = f"Remove outliers with IQR filter? [y/N] (default {'yes' if default_drop else 'no'}): "
    try:
        user_input = input(prompt).strip().lower()
    except EOFError:
        user_input = ""
    if not user_input:
        return default_drop
    if user_input in {"y", "yes", "1", "true", "t"}:
        return True
    if user_input in {"n", "no", "0", "false", "f"}:
        return False
    print(f"Invalid input '{user_input}', using default {'remove' if default_drop else 'keep'}")
    return default_drop


def make_train_test_split(
    X: pd.DataFrame,
    y_log: pd.Series,
    city_series: pd.Series,
    strategy_key: Optional[str] = None,
):
    """Create a train/test split according to the chosen strategy key.

    All split configurations are defined in SPLIT_STRATEGIES and specify
    test-size, random seed and whether the split should be stratified by city.
    Stratifying by city ensures that each city is represented in train and
    test sets with similar proportions, which is important for fair evaluation
    when price levels differ strongly between cities.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used for training and testing.
    y_log : pd.Series
        Log-transformed target values (log1p(realSum)).
    city_series : pd.Series
        City labels corresponding to each row; used for stratification when the
        chosen strategy enables it.
    strategy_key : str, optional
        Key into SPLIT_STRATEGIES. If None, DEFAULT_SPLIT_STRATEGY is used.

    Returns
    -------
    tuple
        (X_train, X_test, y_train_log, y_test_log) as returned by
        sklearn.model_selection.train_test_split.
    """
    key = strategy_key or DEFAULT_SPLIT_STRATEGY
    cfg = SPLIT_STRATEGIES.get(key)
    if cfg is None:
        print(f"Unknown split '{key}', falling back to {DEFAULT_SPLIT_STRATEGY}")
        cfg = SPLIT_STRATEGIES[DEFAULT_SPLIT_STRATEGY]
        key = DEFAULT_SPLIT_STRATEGY
    stratify = city_series if cfg.get("stratify_by_city") else None
    return train_test_split(
        X, y_log,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=stratify,
    )


# ===============================================================
# PREPROCESSOR
# ===============================================================

def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Construct a ColumnTransformer for numeric and categorical features.

    Numeric features are imputed with the median and standardized; categorical
    features (`room_type`, `City`, `metro_dist_bucket`) are imputed with the
    most frequent value and one-hot encoded. This preprocessing pipeline is
    shared across all models so that they are compared on exactly the same
    transformed feature space.

    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame containing all feature columns (without the target).

    Returns
    -------
    ColumnTransformer
        A scikit-learn transformer that can be plugged into a Pipeline before
        the regressor.
    """
    categorical_features = ["room_type", "City", "metro_dist_bucket"]
    numeric_features = [c for c in feature_df.columns if c not in categorical_features]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return preprocessor.get_feature_names_out().tolist()


# ===============================================================
# METRICS & VISUALIZATION
# ===============================================================

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the main regression metrics on the raw price scale.

    The models are trained on log-transformed targets but evaluated after
    back-transformation to Euros. We report RMSE, MAE and R², which together
    capture average error magnitude and explanatory power.
    """
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_pred_vs_actual(y_true, y_pred, model_name, out_path):
    """Scatter plot of predicted vs. actual prices with error coloring.

    Points above the diagonal line represent overpredictions, points below
    represent underpredictions. Colors (red/green) make this visually
    distinguishable. This diagnostic helps to see systematic bias and whether
    the model behaves differently for low vs. high prices.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
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
    ax.set_xlabel("Actual price (€)")
    ax.set_ylabel("Predicted price (€)")
    ax.set_title(f"Predicted vs Actual (raw scale) – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name, out_path):
    """Plot a histogram of residuals (actual - predicted) on the raw scale.

    A roughly symmetric, centered distribution around zero indicates that the
    model is reasonably well calibrated. Strong skewness or heavy tails point
    to systematic under- or overestimation for certain price ranges.

    In the context of this project, these residual plots are meant to be used
    directly in the report or presentation. They visually show how large the
    typical prediction errors are and whether the model systematically under-
    or overestimates certain listings. After removing extreme outliers, the
    residual distribution should become more concentrated around zero with
    fewer very large errors, which is exactly what we want to demonstrate.
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
    """Visualize the top feature importances for tree-based models.

    The input DataFrame is expected to contain `feature` and `importance`
    columns. This plot makes it easy to explain which variables drive the
    model's predictions and to relate them back to the business context.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("importance")
    ax.barh(df_sorted["feature"], df_sorted["importance"])
    ax.set_title(f"Top {len(df)} Features – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def feature_importance_df(pipeline: Pipeline, top_n: int = 10):
    """Extract top-n feature importances from a fitted pipeline, if available.

    Only tree-based models such as RandomForest, XGBoost, LightGBM and CatBoost
    expose a `feature_importances_` attribute. For linear models this function
    returns None. Feature names are taken from the preprocessor so that the
    importances can be matched back to human-readable input variables.
    """
    reg = pipeline.named_steps["regressor"]
    if not hasattr(reg, "feature_importances_"):
        return None
    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
    importances = pd.Series(reg.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    return top.reset_index().rename(columns={"index": "feature", 0: "importance"})


# ===============================================================
# OUTLIER HANDLING
# ===============================================================

def remove_outliers_iqr(df: pd.DataFrame, col: str = "realSum", k: float = 1.5) -> pd.DataFrame:
    """Remove target outliers using the IQR rule.

    Outliers are defined as values below Q1 - k * IQR or above Q3 + k * IQR,
    where Q1 and Q3 are the first and third quartiles of the chosen column.
    For this project we apply the filter to `realSum` to reduce the influence
    of extremely expensive or extremely cheap listings that would otherwise
    distort the error metrics and the model fit.

    The function also prints how many rows are removed and what percentage of
    the dataset this corresponds to. These statistics are useful when writing
    the report or slides: you can directly quote how many observations were
    dropped as outliers and argue that the remaining data better reflects the
    typical price range in the market.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column.
    col : str
        Name of the column on which to compute the IQR-based bounds.
    k : float
        Multiplier for the IQR (1.5 is the classical Tukey rule).

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with outliers removed and index reset.
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
    removed = (~mask).sum()
    print(
        f"Outlier removal on '{col}': "
        f"IQR={iqr:.2f}, lower={lower:.2f}, upper={upper:.2f}. "
        f"Removed {removed} of {len(df)} rows ({removed / len(df) * 100:.2f}%)."
    )

    return df.loc[mask].reset_index(drop=True)


# ===============================================================
# MAIN
# ===============================================================

def main(split_strategy: Optional[str], drop_outliers: Optional[bool], prompt_for_split: bool, prompt_for_outliers: bool) -> None:
    """End-to-end entry point: load data, preprocess, train and evaluate models.

    This function ties together all steps of the project pipeline:
    - optional outlier removal via IQR on the price target,
    - feature engineering for space, distance and safety related signals,
    - configurable train/test split (including city-stratified setups),
    - shared preprocessing (imputation, scaling, one-hot encoding),
    - training multiple regression models on the log-transformed target,
    - back-transformation to Euros and evaluation with RMSE/MAE/R²,
    - creation of plots and model artifacts for documentation and reporting.

    The behavior can be controlled via command-line flags or interactive
    prompts, which makes the script usable both in automated runs and
    interactive exploration.
    """
    load_libomp_if_available()
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    # Choices via prompt if not provided
    if prompt_for_outliers and drop_outliers is None:
        drop_outliers = prompt_outlier_choice()
    drop_outliers = bool(drop_outliers) if drop_outliers is not None else False

    if prompt_for_split and not split_strategy:
        split_strategy = prompt_split_strategy()
    split_strategy = split_strategy or DEFAULT_SPLIT_STRATEGY

    folders = OUTLIER_FOLDERS[drop_outliers]
    PLOTS_DIR = folders["plots"]
    MODELS_DIR = folders["models"]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if drop_outliers:
        df = remove_outliers_iqr(df, col="realSum", k=1.5)

    # -------- Feature engineering on existing columns --------
    df["beds_per_person"] = (df["bedrooms"] / df["person_capacity"]).replace([np.inf, -np.inf], np.nan)
    df["capacity_per_bedroom"] = df["person_capacity"] / (df["bedrooms"] + 1)
    df["capacity_gt2"] = (df["person_capacity"] >= 3).astype(int)
    df["is_studio"] = (df["bedrooms"] == 0).astype(int)
    df["log_metro_dist"] = np.log1p(df["metro_dist"])
    df["log_dist_center"] = np.log1p(df["dist"])
    df["amenity_score"] = (df["attr_index_norm"] + df["rest_index_norm"]) / 2
    if "Safety_Index" in df.columns:
        df["net_safety_score"] = df["Safety_Index"] - df["Crime_Index"]
    else:
        df["net_safety_score"] = -df["Crime_Index"]

    super_raw = df["host_is_superhost"].astype(str).str.strip().str.lower()
    super_map = {"t": 1, "true": 1, "y": 1, "yes": 1, "1": 1}
    df["host_is_superhost"] = super_raw.map(super_map).fillna(0).astype(int)

    max_dist = df["metro_dist"].max()
    if pd.isna(max_dist) or max_dist <= 2:
        max_dist = 2.0001
    df["metro_dist_bucket"] = pd.cut(
        df["metro_dist"],
        bins=[-0.01, 0.5, 2, max_dist],
        labels=["near", "mid", "far"],
        include_lowest=True,
    )

    # Targets
    y_raw = df["realSum"]
    y_log = np.log1p(y_raw)
    X = df.drop(columns=["realSum"])

    # Train/Test split (configurable)
    print(f"Split strategy: {split_strategy}")
    print(f"Outlier removal: {'ON (IQR 1.5)' if drop_outliers else 'OFF'}")
    X_train, X_test, y_train_log, y_test_log = make_train_test_split(
        X, y_log, df["City"], strategy_key=split_strategy
    )

    y_train_raw = np.expm1(y_train_log)
    y_test_raw = np.expm1(y_test_log)

    preprocessor = build_preprocessor(X)

    print(f"Loaded {len(df):,} rows.")
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    model_factories = {
        "Linear Regression": lambda: LinearRegression(),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": lambda: XGBRegressor(
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
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            objective="regression",
            n_jobs=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            n_estimators=400,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            verbose=False,
            random_seed=RANDOM_STATE,
        ),
    }

    results = []
    all_city_scores: Dict[str, pd.DataFrame] = {}

    for model_name, factory in model_factories.items():
        print(f"\n--- Training {model_name} ---")
        reg = factory()

        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", reg),
        ])

        pipe.fit(X_train, y_train_log)

        preds_log = pipe.predict(X_test)
        # Clip predictions in log-space to avoid exploding or negative raw prices.
        # Lower bound 0 → expm1(0) = 0 Euro, upper bound 20 ≈ 4.85 million Euro.
        preds_log = np.clip(preds_log, 0, 20)
        preds_raw = np.expm1(preds_log)

        metrics = eval_metrics(y_test_raw, preds_raw)
        results.append({"Model": model_name, **metrics})

        model_path = MODELS_DIR / f"model_{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved model → {model_path}")

        plot_pred_vs_actual(y_test_raw, preds_raw, model_name,
                            PLOTS_DIR / f"pred_vs_actual_{model_name.replace(' ', '_').lower()}.png")
        plot_residuals(y_test_raw, preds_raw, model_name,
                       PLOTS_DIR / f"residuals_{model_name.replace(' ', '_').lower()}.png")

        fi = feature_importance_df(pipe, top_n=10)
        if fi is not None:
            fi_path = PLOTS_DIR / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            plot_feature_importance(fi, model_name, fi_path)
            print(f"Top 10 Features for {model_name}:\n{fi}")

        test_df = X_test.copy()
        test_df["y_true"] = y_test_raw
        test_df["y_pred"] = preds_raw

        city_scores = (
            test_df.groupby("City")
            .apply(
                lambda g: pd.Series({
                    "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
                    "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                    "R2": r2_score(g["y_true"], g["y_pred"]),
                }),
                include_groups=False,
            )
        )

        print(f"\nPer-City Performance – {model_name}")
        print(city_scores.sort_values("MAE"))

        all_city_scores[model_name] = city_scores

        if model_name == "Decision Tree":
            fig, ax = plt.subplots(figsize=(20, 10))
            feature_names = get_feature_names(pipe.named_steps["preprocessor"])
            plot_tree(
                pipe.named_steps["regressor"],
                feature_names=feature_names,
                filled=True,
                max_depth=3,
                ax=ax,
            )
            tree_path = PLOTS_DIR / "decision_tree_structure.png"
            fig.savefig(tree_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved decision tree visualization → {tree_path}")

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n=== FINAL MODEL PERFORMANCE (RAW SCALE) ===")
    print(results_df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    results_df.plot(x="Model", y="RMSE", kind="bar", ax=axes[0], legend=False)
    axes[0].set_title("RMSE by Model")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=45)

    results_df.plot(x="Model", y="MAE", kind="bar", ax=axes[1], legend=False)
    axes[1].set_title("MAE by Model")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=45)

    results_df.plot(x="Model", y="R2", kind="bar", ax=axes[2], legend=False)
    axes[2].set_title("$R^2$ by Model")
    axes[2].set_ylabel("$R^2$")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    overall_path = PLOTS_DIR / "model_comparison_overall.png"
    fig.savefig(overall_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall model comparison plot → {overall_path}")

    if all_city_scores:
        r2_frames = []
        for model_name, scores in all_city_scores.items():
            if "R2" in scores.columns:
                s = scores["R2"].rename(model_name)
                r2_frames.append(s)

        if r2_frames:
            r2_df = pd.concat(r2_frames, axis=1)

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
            heatmap_path = PLOTS_DIR / "r2_heatmap_city_model.png"
            fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved R^2 heatmap by city and model → {heatmap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Airbnb price models with optional outlier removal.")
    parser.add_argument(
        "--split",
        choices=list(SPLIT_STRATEGIES.keys()),
        help="Choose the train/test split strategy.",
    )
    parser.add_argument(
        "--drop-outliers",
        choices=["yes", "no"],
        help="Remove target outliers using IQR rule before training.",
    )
    args = parser.parse_args()

    drop_flag = None
    if args.drop_outliers == "yes":
        drop_flag = True
    elif args.drop_outliers == "no":
        drop_flag = False

    main(
        split_strategy=args.split,
        drop_outliers=drop_flag,
        prompt_for_split=args.split is None,
        prompt_for_outliers=args.drop_outliers is None,
    )
