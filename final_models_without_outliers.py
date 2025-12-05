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

DATA_PATH = Path("FinalDataSet_geo_merged.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
PLOTS_DIR = Path("plots_without_outliers")
MODELS_DIR = Path("models_without_outliers")


# ===============================================================
# OPTIONAL LIBOMP FOR LIGHTGBM ON MACOS
# ===============================================================

def load_libomp_if_available() -> None:
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        "Crime Index": "Crime_Index",
    })
    df = df.drop(columns=["geo_id"], errors="ignore")
    return df


# ===============================================================
# PREPROCESSOR
# ===============================================================

def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
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
    return preprocessor.get_feature_names_out().tolist()


# ===============================================================
# METRICS & VISUALIZATION
# ===============================================================

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_pred_vs_actual(y_true, y_pred, model_name, out_path):
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
    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("importance")
    ax.barh(df_sorted["feature"], df_sorted["importance"])
    ax.set_title(f"Top {len(df)} Features – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def feature_importance_df(pipeline: Pipeline, top_n: int = 10):
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
    """
    Remove outliers from the DataFrame based on the IQR rule for a single column.

    Rows with values outside [Q1 - k*IQR, Q3 + k*IQR] are dropped.
    Default k=1.5 corresponds to the standard Tukey rule.
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

def main() -> None:
    load_libomp_if_available()
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    df = load_data()
    df = remove_outliers_iqr(df, col="realSum", k=1.5)

    # Targets
    y_raw = df["realSum"]
    y_log = np.log1p(y_raw)
    X = df.drop(columns=["realSum"])

    # Stratified Train/Test split by city
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["City"],
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

    PLOTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

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
    main()