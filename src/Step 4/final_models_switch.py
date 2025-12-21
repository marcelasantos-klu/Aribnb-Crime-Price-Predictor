""" 
Step 4 experiment runner (orchestration script).

What this file does
-------------------
This script is the *single entry point* for running all Step-4 experiments.
It does not implement the ML logic itself; instead, it coordinates the runs and
organises outputs on disk.

Experiments executed
--------------------
1) Regression (main task)
   - Calls `train_for_split(...)` for each configured split strategy.
   - Collects per-model metrics returned by `train_for_split(...)`.
   - Writes an aggregated ranking CSV across split strategies.

2) Classification (complementary task)
   - Calls `run_classification_experiment(...)` once per outlier-setting.

3) Clustering (complementary task)
   - Calls `run_clustering_experiment(...)` once per outlier-setting.

Outlier sensitivity
-------------------
All experiments are run twice:
- drop_flag = False  -> no outlier filtering
- drop_flag = True   -> IQR-based outlier filtering

Important: The actual outlier filtering is implemented in the downstream
modules; this file only toggles the setting and routes outputs.

Output structure
----------------
Artifacts (plots/models/CSVs) are written under `plots&models/` in folders that
encode the chosen outlier-setting and split strategy, e.g.

plots&models/
  <outlier_setting>/
    <split_key>/
      plots/
      models/
    classification/
      plots/
      models/
    clustering/
      plots/

Logging
-------
All console output is captured via `Tee()` and written to `LOG_PATH` so a run can
be reproduced and debugged later.
"""
from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path
from typing import List

import pandas as pd

from config import LOG_PATH, OUTLIER_BASE, SPLIT_STRATEGIES
from data_io import load_data, Tee
from models_regression import train_for_split
from models_classification import run_classification_experiment
from models_clustering import run_clustering_experiment


def load_libomp_if_available() -> None:
    """Try to load libomp on macOS.

    Why this exists:
    - Some gradient-boosting libraries (e.g., LightGBM) require OpenMP.
    - On macOS, missing `libomp` can cause runtime linking errors.

    What we do:
    - Check common installation paths for `libomp.dylib`.
    - If found, load it via `ctypes` and set dynamic linker environment variables.
    - If not found, print a warning (the rest of the pipeline can still run).
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


def main() -> None:
    """Run the full Step-4 pipeline.

    This function coordinates:
    - loading the dataset once,
    - running regression for each split strategy,
    - aggregating regression metrics into a ranking CSV,
    - running classification and clustering experiments,
    - saving plots/models/CSVs into a consistent folder structure,
    - writing a full run log to `LOG_PATH`.

    No modelling logic is implemented here; it lives in the imported `models_*` modules.
    """
    # macOS helper: avoid OpenMP linking issues for some model libraries
    load_libomp_if_available()

    # Ensure local module imports work when running this file directly
    this_dir = Path(__file__).parent
    if str(this_dir) not in sys.path:
        sys.path.append(str(this_dir))

    # Create log directory early so logging never fails due to missing folders
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Capture stdout so the full run can be written to a reproducible log file
    orig_stdout = sys.stdout
    tee = Tee()
    sys.stdout = tee

    # Run both scopes: full data and IQR-filtered “typical-case” scope
    drop_options = [False, True]
    # Load dataset once; downstream modules handle train/test splitting and feature fitting
    df = load_data()

    try:
        for drop_flag in drop_options:
            # Choose the output base folder for this outlier-setting (configured in config.py)
            base_dir = OUTLIER_BASE[drop_flag]
            split_results: List[pd.DataFrame] = []

            # Regression: run for each split strategy (e.g., different train/test definitions)
            for split_key in SPLIT_STRATEGIES.keys():
                split_dir = base_dir / split_key
                plots_dir = split_dir / "plots"
                models_dir = split_dir / "models"
                plots_dir.mkdir(parents=True, exist_ok=True)
                models_dir.mkdir(parents=True, exist_ok=True)
                # Train/evaluate all regression models for this split; save artifacts into plots_dir/models_dir
                res_df = train_for_split(df, split_key, drop_flag, plots_dir, models_dir)
                split_results.append(res_df.assign(Split=split_key))

            # Aggregate regression results across splits to compare models more robustly
            if split_results:
                all_res = pd.concat(split_results, ignore_index=True)
                agg = (
                    all_res.groupby("Model")
                    .agg(
                        RMSE_mean=("RMSE", "mean"),
                        RMSE_std=("RMSE", "std"),
                        MAE_mean=("MAE", "mean"),
                        MAE_std=("MAE", "std"),
                        R2_mean=("R2", "mean"),
                        R2_std=("R2", "std"),
                        Train_RMSE_mean=("Train_RMSE", "mean") if "Train_RMSE" in all_res else ("RMSE", "mean"),
                    )
                    .reset_index()
                    .sort_values("RMSE_mean")
                )
                agg_filename = "regression_ranking_single_split.csv" if len(split_results) == 1 else "regression_ranking_aggregated_across_splits.csv"
                agg_path = base_dir / agg_filename
                agg.to_csv(agg_path, index=False)
                print(f"Saved regression ranking → {agg_path}")

            class_dir = base_dir / "classification"
            clust_dir = base_dir / "clustering"
            class_plots = class_dir / "plots"
            class_models = class_dir / "models"
            clust_plots = clust_dir / "plots"
            class_plots.mkdir(parents=True, exist_ok=True)
            class_models.mkdir(parents=True, exist_ok=True)
            clust_plots.mkdir(parents=True, exist_ok=True)

            # Classification: run once per outlier-setting; saves metrics/plots/models into classification/
            run_classification_experiment(df, drop_flag, class_plots, class_models)
            # Clustering: run once per outlier-setting; saves plots/summaries into clustering/
            run_clustering_experiment(df, drop_flag, clust_plots)

    finally:
        # Always restore stdout and persist the captured log, even if a run crashes
        sys.stdout = orig_stdout
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(tee.getvalue())
        print(f"Saved full training log → {LOG_PATH}")


if __name__ == "__main__":
    main()
