"""
Performance Models: Train and compare five regressors on one source dataset.

Dataset:
- FinalDataSet_geo_merged.csv (realSum in original scale)

Models:
- Linear Regression, Decision Tree, Random Forest, LightGBM, CatBoost

Behavior:
- Cleans columns (Crime_Index/Safety_Index), drops geo_id.
- Uses two targets from the same dataset:
  - raw: realSum as-is
  - log: log1p(realSum), predictions reverted with expm1 for evaluation
- Outputs a single performance table (RMSE/MAE/R2) sorted by RMSE.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
RAW_PATH = Path("FinalDataSet_geo_merged.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and drop identifiers."""
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


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, R2 on raw scale."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def prepare_dataset(path: Path, label: str) -> Dict[str, object]:
    """Load/clean dataset and prepare raw + log targets."""
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    df = pd.read_csv(path)
    df = clean_dataframe(df)

    X = df.drop(columns=["realSum"])
    if "realSum_log" in X.columns:
        X = X.drop(columns=["realSum_log"])

    y_raw = df["realSum"]
    y_log = np.log1p(y_raw)
    targets = [
        {"name": "raw", "y": y_raw, "needs_expm1": False},
        {"name": "log", "y": y_log, "needs_expm1": True},
    ]

    preprocessor_template = build_preprocessor(X)
    return {
        "label": label,
        "X": X,
        "y_true_raw": y_raw,
        "targets": targets,
        "preprocessor": preprocessor_template,
    }


def train_on_dataset(
    dataset: Dict[str, object],
    model_factories: Dict[str, Callable[[], object]],
) -> List[Dict[str, object]]:
    """Train/evaluate all models on all targets for a dataset."""
    X = dataset["X"]
    y_true_raw = dataset["y_true_raw"]
    targets = dataset["targets"]
    preprocessor_template = dataset["preprocessor"]
    label = dataset["label"]

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    results: List[Dict[str, object]] = []

    for model_name, factory in model_factories.items():
        for target in targets:
            reg = factory()
            pipe = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor_template)),
                    ("regressor", reg),
                ]
            )

            y_target = target["y"]
            y_train = y_target.iloc[train_idx]
            y_test_target = y_target.iloc[test_idx]

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            if target["needs_expm1"]:
                preds = np.clip(preds, a_min=None, a_max=20)
                preds_raw = np.expm1(preds)
            else:
                preds_raw = preds

            y_true_eval = y_true_raw.iloc[test_idx].to_numpy()
            preds_raw = np.asarray(preds_raw)
            finite_mask = np.isfinite(y_true_eval) & np.isfinite(preds_raw)
            if not finite_mask.all():
                y_true_eval = y_true_eval[finite_mask]
                preds_raw = preds_raw[finite_mask]

            metrics = eval_metrics(y_true_eval, preds_raw)
            results.append(
                {
                    "Model": model_name,
                    "Dataset": label,
                    "Target": target["name"],
                    **metrics,
                }
            )

    return results


def main() -> None:
    load_libomp_if_available()
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    dataset_raw = prepare_dataset(RAW_PATH, label="raw_file")

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

    all_results: List[Dict[str, object]] = []
    for dataset in [dataset_raw]:
        print(f"\n=== Training on dataset: {dataset['label']} ===")
        all_results.extend(train_on_dataset(dataset, model_factories))

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    print("\n=== Model performance (sorted by RMSE, lower is better) ===")
    print(results_df[["Model", "Dataset", "Target", "RMSE", "MAE", "R2"]])


if __name__ == "__main__":
    main()
