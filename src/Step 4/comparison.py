from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = Path("data/FinalFile/FinalDataSet_geo_merged.csv")
RANDOM_STATE = 42

# Ordner nur für Vergleichsergebnisse
PLOTS_DIR = Path("plots&models/Comparison/plots")
MODELS_DIR = Path("plots&models/Comparison/models")


# ======================================================
# Helper: libomp für LightGBM / XGBoost (gleich wie im Hauptskript)
# ======================================================

def load_libomp_if_available() -> None:
    candidates = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
        Path.home()
        / "Library"
        / "Python"
        / "3.9"
        / "lib"
        / "python"
        / "site-packages"
        / "torch"
        / "lib"
        / "libomp.dylib",
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


# ======================================================
# Data loading, outlier removal, feature engineering
# ======================================================

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Crime Index": "Crime_Index"})
    df = df.drop(columns=["geo_id"], errors="ignore")
    return df


def remove_outliers_iqr(df: pd.DataFrame, target_col: str = "realSum") -> pd.DataFrame:
    """IQR-basierte Outlier-Entfernung auf dem Target."""
    y = df[target_col]
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (y >= lower) & (y <= upper)
    removed = (~mask).sum()
    print(
        f"Outlier removal on '{target_col}': "
        f"Removed {removed} of {len(df)} rows ({removed / len(df) * 100:.2f}%)."
    )
    return df.loc[mask].reset_index(drop=True)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature Engineering wie in deinem final_models.py (WithOutliers)."""
    df = df.copy()

    # Room/space efficiency features
    df["beds_per_person"] = (df["bedrooms"] / df["person_capacity"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df["capacity_per_bedroom"] = df["person_capacity"] / (df["bedrooms"] + 1)
    df["capacity_gt2"] = (df["person_capacity"] >= 3).astype(int)
    df["is_studio"] = (df["bedrooms"] == 0).astype(int)

    # Distance-based transformations
    df["log_metro_dist"] = np.log1p(df["metro_dist"])
    df["log_dist_center"] = np.log1p(df["dist"])

    # Amenity / location quality scores
    df["amenity_score"] = (df["attr_index_norm"] + df["rest_index_norm"]) / 2
    if "Safety_Index" in df.columns:
        df["net_safety_score"] = df["Safety_Index"] - df["Crime_Index"]
    else:
        df["net_safety_score"] = -df["Crime_Index"]

    # Robust encoding of host_is_superhost
    if "host_is_superhost" in df.columns:
        super_raw = df["host_is_superhost"].astype(str).str.strip().str.lower()
        super_map = {"t": 1, "true": 1, "y": 1, "yes": 1, "1": 1}
        df["host_is_superhost"] = super_raw.map(super_map).fillna(0).astype(int)

    # Metro buckets (für Vergleich verwenden wir sie nur in FE-Variante)
    max_dist = df["metro_dist"].max()
    if pd.isna(max_dist) or max_dist <= 2:
        max_dist = 2.0001
    df["metro_dist_bucket"] = pd.cut(
        df["metro_dist"],
        bins=[-0.01, 0.5, 2, max_dist],
        labels=["near", "mid", "far"],
        include_lowest=True,
    )

    return df


# ======================================================
# Preprocessor & Metrics
# ======================================================

def build_preprocessor(feature_df: pd.DataFrame, use_buckets: bool) -> ColumnTransformer:
    if use_buckets:
        categorical_features = ["room_type", "City", "metro_dist_bucket"]
    else:
        categorical_features = ["room_type", "City"]
    numeric_features = [c for c in feature_df.columns if c not in categorical_features]

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ======================================================
# Kernel: Ein Experiment laufen lassen
# ======================================================

def run_experiment(
    df: pd.DataFrame,
    label: str,
    use_feature_engineering: bool,
    use_outliers: bool,
) -> pd.DataFrame:
    """
    Trainiere alle Modelle für einen bestimmten Setup und gib Metrics als DataFrame zurück.
    """
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    work_df = df.copy()

    if use_outliers:
        # With outliers = nichts tun
        pass
    else:
        work_df = remove_outliers_iqr(work_df, target_col="realSum")

    if use_feature_engineering:
        work_df = apply_feature_engineering(work_df)
        use_buckets = True
    else:
        # Falls metro_dist_bucket noch vom vorherigen Lauf übrig sein sollte
        work_df = work_df.drop(columns=["metro_dist_bucket"], errors="ignore")
        use_buckets = False

    # Target & Features
    y_raw = work_df["realSum"]
    y_log = np.log1p(y_raw)
    X = work_df.drop(columns=["realSum"])

    # Einheitlicher Split: City-stratified 80/20, Seed 42 (fix)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=42,
        stratify=work_df["City"],
    )
    y_test_raw = np.expm1(y_test_log)

    preprocessor = build_preprocessor(X, use_buckets=use_buckets)

    model_factories = {
        "Linear Regression": lambda: LinearRegression(),
        "Decision Tree": lambda: DecisionTreeRegressor(
            max_depth=12, random_state=RANDOM_STATE
        ),
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

    rows = []
    for model_name, factory in model_factories.items():
        print(f"[{label}] Training {model_name} (FE={use_feature_engineering}, Outliers={'keep' if use_outliers else 'removed'})")
        reg = factory()
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor(X, use_buckets=use_buckets)),
                ("regressor", reg),
            ]
        )
        pipe.fit(X_train, y_train_log)
        preds_log = pipe.predict(X_test)
        preds_log = np.clip(preds_log, 0, 20)
        preds_raw = np.expm1(preds_log)

        m = eval_metrics(y_test_raw, preds_raw)
        rows.append(
            {
                "Variant": label,
                "Use_FE": use_feature_engineering,
                "Use_Outliers": use_outliers,
                "Model": model_name,
                **m,
            }
        )

        # Optional: Modelle speichern, falls du willst
        model_file = (
            MODELS_DIR
            / f"{label}_{'fe' if use_feature_engineering else 'base'}_{model_name.replace(' ', '_').lower()}.pkl"
        )
        joblib.dump(pipe, model_file)

    return pd.DataFrame(rows)


# ======================================================
# Main: alle Varianten in einem Rutsch
# ======================================================

def main() -> None:
    load_libomp_if_available()
    df = load_data()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # A: with outliers, base features
    all_results.append(
        run_experiment(df, label="with_outliers_base", use_feature_engineering=False, use_outliers=True)
    )
    # B: with outliers, feature engineering
    all_results.append(
        run_experiment(df, label="with_outliers_fe", use_feature_engineering=True, use_outliers=True)
    )
    # C: without outliers, base features
    all_results.append(
        run_experiment(df, label="no_outliers_base", use_feature_engineering=False, use_outliers=False)
    )
    # D: without outliers, feature engineering
    all_results.append(
        run_experiment(df, label="no_outliers_fe", use_feature_engineering=True, use_outliers=False)
    )

    results_df = pd.concat(all_results, ignore_index=True)

    print("\n=== COMBINED COMPARISON (ALL VARIANTS & MODELS) ===")
    print(results_df.sort_values(["Model", "Variant"]))

    # Optional: Summary-Pivot für RMSE
    pivot_rmse = results_df.pivot_table(
        index=["Model"],
        columns=["Variant"],
        values="RMSE",
    )
    print("\n=== RMSE by Model and Variant ===")
    print(pivot_rmse)

    # Kleines Plot-Beispiel: RMSE-Vergleich für ein Modell-Set (z.B. Random Forest)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model_name in results_df["Model"].unique():
        subset = results_df[results_df["Model"] == model_name]
        ax.plot(
            subset["Variant"],
            subset["RMSE"],
            marker="o",
            label=model_name,
        )
    ax.set_title("RMSE across Variants (all Models)")
    ax.set_ylabel("RMSE")
    ax.set_xticklabels(results_df["Variant"].unique(), rotation=45)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "rmse_comparison_variants.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Benchmark-Plot: durchschnittlicher RMSE pro Variant (Outliers/FE-Szenarien)
    benchmark_rmse = (
        results_df.groupby("Variant")["RMSE"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(benchmark_rmse.index, benchmark_rmse.values)
    ax.set_title("Benchmark: Average RMSE by Variant")
    ax.set_ylabel("Average RMSE")
    ax.set_xlabel("Variant")
    ax.set_xticklabels(benchmark_rmse.index, rotation=45)

    # Kleine Wertebeschriftung auf den Balken, damit es im Report klar lesbar ist
    for i, (variant, rmse_val) in enumerate(benchmark_rmse.items()):
        ax.text(
            i,
            rmse_val,
            f"{rmse_val:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "benchmark_rmse_by_variant.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()