"""
Train, evaluate, visualize, and save five regression models on Airbnb pricing data.

Dataset:
- FinalDataSet_geo_merged.csv (realSum in euro; target)

Models (log-target for final training):
- Linear Regression (OLS)
- Decision Tree Regressor
- Random Forest Regressor
- LightGBM Regressor
- CatBoost Regressor

Behavior:
- Cleans columns (Crime_Index/Safety_Index), drops geo_id.
- Uses two targets from the same dataset for evaluation:
  - raw: realSum
  - log: log1p(realSum) (predictions converted back with expm1 for metrics/plots)
- Final models are trained on the log target and saved as joblib pickles.
- Generates per-model plots: predicted vs actual, residuals; feature importances for tree models.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --- Configuration ---
DATA_PATH = Path("FinalDataSet_geo_merged.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
PLOTS_DIR = Path("plots")
MODELS_DIR = Path("models")


def load_libomp_if_available() -> Optional[Path]:
    """Load libomp if present (needed for LightGBM on macOS without Homebrew)."""
    candidates = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
    ]
    torch_lib = (
        Path.home()
        / "Library"
        / "Python"
        / "3.9"
        / "lib"
        / "python"
        / "site-packages"
        / "torch"
        / "lib"
        / "libomp.dylib"
    )
    candidates.append(torch_lib)
    for path in candidates:
        if path.exists():
            try:
                ctypes.cdll.LoadLibrary(str(path))
                os.environ.setdefault("DYLD_LIBRARY_PATH", str(path.parent))
                print(f"Loaded libomp from: {path}")
                return path
            except OSError:
                continue
    print("libomp not found; LightGBM may fail to load.")
    return None


def load_data() -> pd.DataFrame:
    """Load and clean dataset: rename columns, drop geo_id."""
    if not DATA_PATH.exists():
        raise SystemExit(f"Input file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Crime Index": "Crime_Index", "Safety Index": "Safety_Index"})
    df = df.drop(columns=["geo_id"], errors="ignore")
    return df


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer with scaling for numeric and one-hot for categorical."""
    categorical_features = ["room_type", "City"]
    numeric_features = [c for c in feature_df.columns if c not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract feature names after preprocessing for importance reporting."""
    return preprocessor.get_feature_names_out().tolist()


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, R2 on raw scale."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_path: Path) -> None:
    """Scatter plot of predicted vs actual values with y=x reference."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#4C72B0")
    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual realSum")
    ax.set_ylabel("Predicted realSum")
    ax.set_title(f"Predicted vs Actual - {model_name} (log target)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_path: Path) -> None:
    """Histogram of residuals (actual - predicted)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40, color="#55A868", alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Residuals - {model_name} (log target)")
    ax.set_xlabel("Actual - Predicted (euros)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Horizontal bar plot for feature importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("importance")
    ax.barh(df_sorted["feature"], df_sorted["importance"], color="#C44E52")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(df)} features - {model_name} (log target)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def feature_importance_df(pipeline: Pipeline, top_n: int = 10) -> Optional[pd.DataFrame]:
    """Return top-N feature importances as a DataFrame if available."""
    reg = pipeline.named_steps["regressor"]
    if not hasattr(reg, "feature_importances_"):
        return None
    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
    importances = pd.Series(reg.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    return top.reset_index().rename(columns={"index": "feature", 0: "importance"})


def main() -> None:
    # Ensure LightGBM can link.
    load_libomp_if_available()
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    # --- Load and prepare data ---
    df = load_data()
    y_raw = df["realSum"]
    y_log = np.log1p(y_raw)
    X = df.drop(columns=["realSum"])

    preprocessor = build_preprocessor(X)

    # Shared split for all models
    X_train, X_test, y_train_log, y_test_raw = train_test_split(
        X,
        y_log,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=None,
    )
    # Align raw target with the same split
    y_test_raw = y_raw.loc[y_test_raw.index] if hasattr(y_test_raw, "index") else y_test_raw
    print(f"Loaded {len(df):,} rows. Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    # --- Define models (final versions trained on log target) ---
    model_factories: Dict[str, Callable[[], object]] = {
        "Linear Regression": lambda: LinearRegression(),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            n_estimators=400,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
    }

    results: List[Dict[str, object]] = []
    PLOTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    for model_name, factory in model_factories.items():
        print(f"\n--- Training final log model: {model_name} ---")
        reg = factory()
        pipe = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("regressor", reg),
            ]
        )

        # Fit on log target
        pipe.fit(X_train, y_train_log)

        # Predict (log) and revert to raw scale
        preds_log = pipe.predict(X_test)
        preds_log = np.clip(preds_log, a_min=None, a_max=20)  # avoid overflow in expm1
        preds_raw = np.expm1(preds_log)

        # Metrics on raw scale
        metrics = eval_metrics(y_test_raw.to_numpy(), preds_raw)
        results.append({"Model": model_name, **metrics})

        # Save model
        model_path = MODELS_DIR / f"final_model_{model_name.replace(' ', '_').lower()}_log.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved model to: {model_path}")

        # Plots
        pred_plot_path = PLOTS_DIR / f"pred_vs_actual_{model_name.replace(' ', '_').lower()}_log.png"
        resid_plot_path = PLOTS_DIR / f"residuals_{model_name.replace(' ', '_').lower()}_log.png"
        plot_pred_vs_actual(
            y_true=y_test_raw.to_numpy(),
            y_pred=preds_raw,
            model_name=model_name,
            out_path=pred_plot_path,
        )
        plot_residuals(
            y_true=y_test_raw.to_numpy(),
            y_pred=preds_raw,
            model_name=model_name,
            out_path=resid_plot_path,
        )
        print(f"Saved plots: {pred_plot_path.name}, {resid_plot_path.name}")

        if model_name in {"Decision Tree", "Random Forest", "LightGBM", "CatBoost"}:
            fi = feature_importance_df(pipe, top_n=10)
            if fi is not None:
                fi_plot_path = PLOTS_DIR / f"feature_importance_{model_name.replace(' ', '_').lower()}_log.png"
                plot_feature_importance(
                    fi,
                    model_name=model_name,
                    out_path=fi_plot_path,
                )
                print(f"Top 10 features for {model_name}:\n{fi.to_string(index=False)}")
                print(f"Saved feature importance plot: {fi_plot_path.name}")

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    print("\n=== Final log-model performance (sorted by RMSE) ===")
    print(results_df[["Model", "RMSE", "MAE", "R2"]])


if __name__ == "__main__":
    main()
