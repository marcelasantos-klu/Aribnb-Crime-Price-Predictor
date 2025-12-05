"""
Train, evaluate, visualize, and save regression models for Airbnb price prediction
with correct city-aware splitting, log-target training, and raw-scale evaluation.

Changes vs. old version:
- Clean separation of y_log and y_raw in the split.
- Stratified train/test split by city.
- Clear titles in plots (raw-scale).
- Added per-city performance evaluation.
- Removed confusing index-alignment hack.
- Improved documentation.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

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
PLOTS_DIR = Path("plots&models/WithOutliers/plots")
MODELS_DIR = Path("plots&models/WithOutliers/models")
# Add more split options here; switch SPLIT_STRATEGY to pick one.
SPLIT_STRATEGIES = {
    # Current default: 80/20 split, stratified by city
    "city_stratified_80_20_seed42": {"test_size": 0.2, "random_state": 42, "stratify_by_city": True},
    # More conservative hold-out for broader testing
    "city_stratified_70_30_seed42": {"test_size": 0.3, "random_state": 42, "stratify_by_city": True},
    # Same 80/20 but different seed to check stability
    "city_stratified_80_20_seed99": {"test_size": 0.2, "random_state": 99, "stratify_by_city": True},
}
DEFAULT_SPLIT_STRATEGY = "city_stratified_80_20_seed42"
SPLIT_STRATEGY = DEFAULT_SPLIT_STRATEGY


# ===============================================================
# OPTIONAL LIBOMP FOR LIGHTGBM ON MACOS
# ===============================================================

def load_libomp_if_available() -> None:
    """
    Try to load the OpenMP runtime library (libomp) on macOS.

    This is primarily needed for LightGBM (and sometimes XGBoost) on macOS,
    where the OpenMP library is not always available on the default search path.
    The function checks a list of candidate locations (Homebrew, /usr/local, and
    the PyTorch installation path). If libomp is found and successfully loaded,
    the relevant DYLD environment variables are set so that libraries depending
    on OpenMP can find it at runtime.

    The function prints a short status message indicating whether libomp was loaded
    or not. It does not raise an exception if libomp is missing; LightGBM may still
    fail at import time in that case.
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
    """
    Load the Airbnb dataset from disk and perform basic column cleanup.

    - Verifies that the CSV file defined by DATA_PATH exists.
    - Reads the file into a pandas DataFrame.
    - Normalizes the 'Crime Index' column name to 'Crime_Index'.
    - Drops the 'geo_id' column if present (it is only used as a spatial key).

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame containing all features and the target column `realSum`.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Crime Index": "Crime_Index",
    })
    df = df.drop(columns=["geo_id"], errors="ignore")
    return df


def prompt_split_strategy(default: str = DEFAULT_SPLIT_STRATEGY) -> str:
    """
    Ask the user to pick a split strategy at runtime.
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


def make_train_test_split(
    X: pd.DataFrame,
    y_log: pd.Series,
    city_series: pd.Series,
    strategy_key: Optional[str] = None,
):
    """
    Create a train/test split based on the configured SPLIT_STRATEGY.

    Returns
    -------
    tuple
        X_train, X_test, y_train_log, y_test_log
    """
    key = strategy_key or SPLIT_STRATEGY
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
    """
    Build a ColumnTransformer that preprocesses numeric and categorical features.

    Parameters
    ----------
    feature_df : pd.DataFrame
        A sample DataFrame containing the feature columns (excluding the target).
        The column names are used to determine which features are numeric and
        which are categorical.

    Returns
    -------
    ColumnTransformer
        A transformer that:
        - imputes missing numeric values with the median and standardizes them,
        - imputes missing categorical values with the most frequent category and
          one-hot encodes them, ignoring unseen categories at prediction time.

    Notes
    -----
    The categorical features are hard-coded as ['room_type', 'City'].
    All remaining columns are treated as numeric.
    """
    categorical_features = ["room_type", "City"]
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
    """
    Return the expanded feature names after preprocessing.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The fitted ColumnTransformer used in the pipeline.

    Returns
    -------
    List[str]
        A list of feature names corresponding to the transformed design matrix.
        This includes:
        - numeric feature names as-is (prefixed by 'num__'),
        - one-hot encoded categorical feature names (prefixed by 'cat__').
    """
    return preprocessor.get_feature_names_out().tolist()


# ===============================================================
# METRICS & VISUALIZATION
# ===============================================================

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics on the raw target scale.

    Parameters
    ----------
    y_true : np.ndarray
        The ground-truth target values (here: actual prices in Euro).
    y_pred : np.ndarray
        The predicted target values on the same scale.

    Returns
    -------
    Dict[str, float]
        A dictionary containing:
        - 'RMSE': Root Mean Squared Error,
        - 'MAE': Mean Absolute Error,
        - 'R2' : Coefficient of determination.
    """
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_pred_vs_actual(y_true, y_pred, model_name, out_path):
    """
    Create a scatter plot comparing predicted vs actual prices for a given model.

    The plot:
    - uses the actual price on the x-axis and the predicted price on the y-axis,
    - colors points red if the model overpredicts (pred > actual),
      and green if it underpredicts (pred < actual),
    - draws a diagonal reference line (perfect prediction),
    - adds a legend explaining the color coding.

    Parameters
    ----------
    y_true : array-like
        Actual prices in Euro from the test set.
    y_pred : array-like
        Model predictions on the same raw price scale.
    model_name : str
        Name of the model (used in the plot title and output file name).
    out_path : Path or str
        Destination path where the PNG file will be saved.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Color code: red = overprediction, green = underprediction
    colors = ["red" if yp > yt else "green" for yt, yp in zip(y_true, y_pred)]

    ax.scatter(y_true, y_pred, alpha=0.5, s=12, c=colors)

    # Legend: red = overprediction, green = underprediction
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markersize=6, label="Overprediction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="green", markersize=6, label="Underprediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Error type")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

    # Diagonal reference line
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
    """
    Plot a histogram of residuals (actual - predicted) for a given model.

    Parameters
    ----------
    y_true : array-like
        Actual target values on the raw scale.
    y_pred : array-like
        Predicted target values on the same scale.
    model_name : str
        Name of the model (used in the plot title and file name).
    out_path : Path or str
        Destination path for the residual histogram PNG.
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
    """
    Plot a horizontal bar chart of feature importances for a given model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with two columns:
        - 'feature': feature names,
        - 'importance': corresponding importance scores.
    model_name : str
        Name of the model (used in the plot title and file name).
    out_path : Path
        Destination path for the feature importance PNG.

    Notes
    -----
    The DataFrame is assumed to be pre-filtered to the top N features.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("importance")
    ax.barh(df_sorted["feature"], df_sorted["importance"])
    ax.set_title(f"Top {len(df)} Features – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def feature_importance_df(pipeline: Pipeline, top_n: int = 10):
    """
    Extract a DataFrame of the top-N feature importances from a fitted pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline that contains:
        - a 'preprocessor' step (ColumnTransformer),
        - a 'regressor' step with a `feature_importances_` attribute
          (e.g. tree-based models).
    top_n : int, default=10
        Number of most important features to return.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with columns:
        - 'feature': feature name,
        - 'importance': importance score.
        Returns None if the regressor does not expose `feature_importances_`.
    """
    reg = pipeline.named_steps["regressor"]
    if not hasattr(reg, "feature_importances_"):
        return None
    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
    importances = pd.Series(reg.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    return top.reset_index().rename(columns={"index": "feature", 0: "importance"})


# ===============================================================
# MAIN
# ===============================================================

def main(split_strategy: Optional[str] = None, prompt_for_split: bool = False) -> None:
    """
    End-to-end training, evaluation, and visualization pipeline for price models.

    Steps performed:
    1. Load the dataset and basic configuration.
    2. Build a shared preprocessing pipeline for numeric and categorical features.
    3. Split the data into train and test sets, stratified by city.
       The models are trained on the log-transformed target (`log1p(realSum)`),
       but all evaluation is done on the original price scale via `expm1`.
    4. For each model type (Linear Regression, Decision Tree, Random Forest,
       XGBoost, LightGBM, CatBoost):
       - construct a pipeline (preprocessor + regressor),
       - train on the log target,
       - predict on the test set and transform predictions back to the raw scale,
       - compute RMSE, MAE, and R²,
       - save the fitted pipeline to disk,
       - generate diagnostic plots (predicted vs actual, residuals),
       - plot feature importances where available,
       - evaluate performance per city and print a table.
    5. After all models are trained, aggregate the metrics into a summary table,
       generate bar charts comparing models, and create a heatmap of R² by city
       and model.

    This function is intended to be run as a script entry point and will produce:
    - trained model files under MODELS_DIR,
    - multiple PNG plots under PLOTS_DIR,
    - and a textual summary printed to stdout.
    """
    load_libomp_if_available()
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    df = load_data()

    # Targets
    y_raw = df["realSum"]
    y_log = np.log1p(y_raw)
    X = df.drop(columns=["realSum"])

    # Train/Test split (configurable)
    selected_split = split_strategy
    if prompt_for_split and not selected_split:
        selected_split = prompt_split_strategy()
    selected_split = selected_split or SPLIT_STRATEGY
    print(f"Split strategy: {selected_split}")
    X_train, X_test, y_train_log, y_test_log = make_train_test_split(
        X, y_log, df["City"], strategy_key=selected_split
    )

    # Raw-scale targets for evaluation
    y_train_raw = np.expm1(y_train_log)
    y_test_raw = np.expm1(y_test_log)

    preprocessor = build_preprocessor(X)

    print(f"Loaded {len(df):,} rows.")
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # MODELS
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

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    all_city_scores: Dict[str, pd.DataFrame] = {}

    for model_name, factory in model_factories.items():
        print(f"\n--- Training {model_name} ---")
        reg = factory()

        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", reg),
        ])

        # Train (log target)
        pipe.fit(X_train, y_train_log)

        # Predict → log → raw
        preds_log = pipe.predict(X_test)
        preds_log = np.clip(preds_log, None, 20)
        preds_raw = np.expm1(preds_log)

        # Metrics (raw scale)
        metrics = eval_metrics(y_test_raw, preds_raw)
        results.append({"Model": model_name, **metrics})

        # Save model
        model_path = MODELS_DIR / f"model_{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved model → {model_path}")

        # Plots
        plot_pred_vs_actual(y_test_raw, preds_raw, model_name,
                            PLOTS_DIR / f"pred_vs_actual_{model_name.replace(' ', '_').lower()}.png")
        plot_residuals(y_test_raw, preds_raw, model_name,
                       PLOTS_DIR / f"residuals_{model_name.replace(' ', '_').lower()}.png")

        # Feature Importances
        fi = feature_importance_df(pipe, top_n=10)
        if fi is not None:
            fi_path = PLOTS_DIR / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            plot_feature_importance(fi, model_name, fi_path)
            print(f"Top 10 Features for {model_name}:\n{fi}")

        # ---------------------------------
        # PER-CITY EVALUATION
        # ---------------------------------
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
            # Plot a visual representation of the trained decision tree (limited depth for readability)
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

    # FINAL RESULTS
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n=== FINAL MODEL PERFORMANCE (RAW SCALE) ===")
    print(results_df)

    # -------------------------------------------------
    # ADDITIONAL PERFORMANCE PLOTS FOR THE REPORT
    # -------------------------------------------------

    # 1) Bar charts for RMSE, MAE, and R2 per model
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

    # 2) Heatmap of R2 per City and Model (if per-city scores are available)
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
    parser = argparse.ArgumentParser(description="Train Airbnb price models.")
    parser.add_argument(
        "--split",
        choices=list(SPLIT_STRATEGIES.keys()),
        help="Choose the train/test split strategy.",
    )
    args = parser.parse_args()
    main(split_strategy=args.split, prompt_for_split=args.split is None)
