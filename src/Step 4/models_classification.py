"""Step 4 – Classification modelling & evaluation (high-revenue detection).

This module reframes the revenue prediction task as a binary classification problem:
given listing characteristics, city, and surrounding conditions, predict whether a
booking is "high revenue".

Label definition
---------------
- The "high revenue" class is defined using a **train-only threshold**:
  `high = 1 if realSum >= median(realSum_train) else 0`.
  This prevents label leakage from the test set.

Split & leakage prevention
-------------------------
- We use a single 80/20 train/test split, stratified by `City` when possible.
- Optional outlier filtering is applied on the TRAIN set only (IQR rule on `realSum`).
- Feature-engineering parameters are computed on TRAIN and then applied to TEST.

Evaluation metrics
------------------
We report Accuracy, Precision, Recall, and F1 on the test set.
- F1 is the main comparison metric because it balances Precision and Recall.
- Precision indicates how many predicted “high revenue” cases are truly high.
- Recall indicates how many truly “high revenue” cases are recovered.

Interpretation warning
----------------------
This experiment is descriptive and predictive only. It should not be interpreted
as evidence that any feature *causes* high revenue.

Outputs
-------
The experiment writes classifiers and evaluation artifacts into the provided directories:
- `classification_metrics.csv` and `classification_model_comparison.csv`
- `classification_model_comparison.png` (F1/Precision/Recall comparison)
- one confusion-matrix PNG per classifier
- one serialized classifier pipeline (`.pkl`) per model

Note: This is a complementary analysis. It does not replace regression and must not be
interpreted causally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE
from features import compute_fe_params, _feature_engineering_for_ml


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for classification models.

    - Numeric features: median imputation + standard scaling
    - Categorical features: most-frequent imputation + one-hot encoding

    KNN/linear models benefit from scaling; trees are less sensitive, but we keep a
    consistent preprocessing pipeline for fair comparisons.
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


def run_classification_experiment(
    base_df: pd.DataFrame,
    drop_outliers: bool,
    plots_dir: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """Run one full classification experiment (high-revenue vs. other).

    Parameters
    ----------
    base_df:
        Input dataframe containing the target column `realSum` and all features.
    drop_outliers:
        If True, removes TRAIN outliers on `realSum` using an IQR rule (train-only).
    plots_dir / models_dir:
        Output folders for plots/CSVs and serialized model artifacts.

    Returns
    -------
    pandas.DataFrame
        Model comparison table sorted by F1 (descending).
    """
    df = base_df.copy()
    # Work on a copy so the caller's dataframe is never mutated.

    # Separate target (continuous revenue) from raw feature columns.
    y_real = df["realSum"].to_numpy()
    X_raw = df.drop(columns=["realSum"])

    # 80/20 split. City-stratification helps ensure the test set is not dominated by a few cities.
    try:
        X_train_raw, X_test_raw, y_train_real, y_test_real = train_test_split(
            X_raw,
            y_real,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=X_raw["City"],
        )
    except ValueError as e:
        # If some cities are too small, stratification can fail; we fall back to a standard split.
        print(f"Warning: city-stratified split failed ({e}). Falling back to non-stratified split.")
        X_train_raw, X_test_raw, y_train_real, y_test_real = train_test_split(
            X_raw,
            y_real,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=None,
        )

    # Optional: remove extreme revenue cases from TRAIN only (IQR rule). This preserves a clean test set.
    if drop_outliers:
        y_train_s = pd.Series(y_train_real)
        q1 = y_train_s.quantile(0.25)
        q3 = y_train_s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = y_train_s.between(lower, upper)
        removed = int((~mask).sum())
        print(
            f"Train-only outlier removal on 'realSum' (classification): IQR={iqr:.2f}, lower={lower:.2f}, upper={upper:.2f}. "
            f"Removed {removed} of {len(mask)} train rows ({removed / len(mask) * 100:.2f}%)."
        )
        # Apply the mask to TRAIN features/target; reset index to keep downstream processing simple.
        X_train_raw = X_train_raw.loc[mask.values].reset_index(drop=True)
        y_train_real = y_train_s.loc[mask.values].reset_index(drop=True).to_numpy()

    # Define the binary label using a TRAIN-only median threshold to avoid leaking TEST information.
    threshold = float(np.median(y_train_real))
    y_train = (y_train_real >= threshold).astype(int)
    y_test = (y_test_real >= threshold).astype(int)

    # Fit feature-engineering parameters on TRAIN only, then apply the same transformation to TEST.
    fe_params = compute_fe_params(X_train_raw)
    X_train = _feature_engineering_for_ml(X_train_raw, fe_params=fe_params)
    X_test = _feature_engineering_for_ml(X_test_raw, fe_params=fe_params)

    # Build the preprocessing transformer using the engineered TRAIN feature set.
    preprocessor = build_preprocessor(X_train)
    if drop_outliers:
        print("(Train size reflects train-only outlier filtering)")

    # Compare a small set of diverse classifiers to illustrate bias/variance trade-offs.
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results: List[Dict[str, float]] = []

    for name, clf in classifiers.items():
        print(f"\n--- Classification: {name} ---")
        # Each model is a full pipeline: preprocessing -> classifier. This keeps training/evaluation consistent.
        pipe = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("classifier", clf),
            ]
        )

        # Train on TRAIN, then predict on TEST.
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Compute standard classification metrics on the held-out TEST set.
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
            }
        )

        # Persist the fitted pipeline so results can be reproduced without retraining.
        model_path = models_dir / f"classifier_{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved classifier → {model_path}")

        # Confusion matrix helps diagnose the types of errors (false positives vs false negatives).
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Low/Normal", "High"])
        ax.set_yticklabels(["Low/Normal", "High"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion Matrix – {name}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = plots_dir / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        fig.savefig(cm_path, dpi=300)
        plt.close(fig)
        print(f"Saved confusion matrix → {cm_path}")

    # Collect all model results into a comparison table (sorted by F1).
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print("\n=== CLASSIFICATION PERFORMANCE (High-Revenue vs. Other) ===")
    print(results_df)
    results_path = plots_dir / "classification_metrics.csv"
    # Save metrics tables for the report and for quick inspection.
    results_df.to_csv(results_path, index=False)
    print(f"Saved classification metrics → {results_path}")
    comp_path = plots_dir / "classification_model_comparison.csv"
    sorted_results = results_df.sort_values("F1", ascending=False)
    sorted_results.to_csv(comp_path, index=False)
    print(f"Saved classification model comparison table → {comp_path}")

    # Visual comparison of F1/Precision/Recall across models (for the report appendix).
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    sorted_results.plot(x="Model", y="F1", kind="barh", ax=axes[0], legend=False)
    axes[0].set_title("F1 by Model")
    axes[0].set_xlabel("F1")
    axes[0].invert_yaxis()

    sorted_results.plot(x="Model", y="Precision", kind="barh", ax=axes[1], legend=False)
    axes[1].set_title("Precision by Model")
    axes[1].set_xlabel("Precision")
    axes[1].invert_yaxis()

    sorted_results.plot(x="Model", y="Recall", kind="barh", ax=axes[2], legend=False)
    axes[2].set_title("Recall by Model")
    axes[2].set_xlabel("Recall")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plot_path = plots_dir / "classification_model_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved classification comparison plot → {plot_path}")
    return results_df
