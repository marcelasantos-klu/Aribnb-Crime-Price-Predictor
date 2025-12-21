"""
Clustering experiment (K-Means).

Goal
----
This module runs an unsupervised K-Means clustering on engineered listing/location features.
The objective is *exploratory*: identify groups of similar listings and describe how these
clusters differ in terms of location/size/context characteristics.

Critical leakage note
---------------------
`realSum` (revenue) is **NOT** used as an input feature for clustering.
It is used only *afterwards* to summarise the revenue profile of each cluster.

Evaluation criteria
-------------------
We evaluate k in {2, 3, 4} using:
- Inertia (within-cluster sum of squares): used for an elbow-style diagnostic.
- Silhouette score: higher is better; summarises cluster separation and cohesion.

Outlier sensitivity
-------------------
Because clustering is unsupervised (no held-out test split), optional outlier removal is
applied to the full dataset copy when `drop_outliers=True`. This is done only to make
cluster summaries more stable and interpretable.

Interpretation warning
----------------------
Clustering is descriptive. Any association between clusters and revenue should not be
interpreted as causal.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from features import compute_fe_params, _feature_engineering_for_ml, remove_outliers_iqr


def run_clustering_experiment(
    base_df: pd.DataFrame,
    drop_outliers: bool,
    plots_dir: Path,
) -> pd.DataFrame:
    """Run the Step 4 clustering experiment using K-Means.

    We cluster listings based on engineered listing/location features and evaluate k in {2, 3, 4}
    using inertia (elbow) and silhouette score.

    Important for the report
    ------------------------
    - `realSum` (revenue) is **NOT** used as an input feature for clustering. It is used only
      after clustering to summarize the revenue profile of each cluster.
    - Because clustering is unsupervised (no held-out test split), optional outlier removal is
      applied to the full dataset copy when `drop_outliers=True`.

    Parameters
    ----------
    base_df:
        Input dataframe containing listing features and the revenue column `realSum`.
    drop_outliers:
        If True, removes extreme `realSum` values using an IQR rule (for more stable cluster summaries).
    plots_dir:
        Output directory where CSV summaries and PNG plots are written.

    Files written
    ------------
    - `cluster_summary_k{2,3,4}.csv`: per-cluster means/medians for selected features + revenue.
    - `cluster_mean_revenue_k{2,3,4}.png`: mean revenue per cluster (interpretation only).
    - `kmeans_k_diagnostics.csv`: inertia and silhouette for each k.
    - `clustering_model_comparison.csv` + `clustering_model_comparison.png`: silhouette ranking across k.
    - `kmeans_elbow_inertia.png` and `kmeans_silhouette.png`: diagnostic curves across k.

    Returns
    -------
    pandas.DataFrame
        Diagnostics table with columns `k`, `inertia`, and `silhouette`.
    """
    # Ensure the output folder exists for CSV summaries and plots.
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Work on a copy so the caller's dataframe is never mutated.
    df = base_df.copy()
    # Optional: remove extreme revenue cases from the full dataset copy (unsupervised setting).
    if drop_outliers:
        df = remove_outliers_iqr(df, col="realSum", k=1.5)

    # Fit feature-engineering parameters on the (possibly filtered) dataset, then transform features.
    fe_params = compute_fe_params(df)
    df = _feature_engineering_for_ml(df, fe_params=fe_params)

    # Select clustering inputs (exclude `realSum` on purpose to avoid target leakage).
    cluster_features = [
        "person_capacity",
        "bedrooms",
        "amenity_score",
        "log_metro_dist",
        "log_dist_center",
        "Crime_Index",
    ]
    # Extract the clustering feature matrix.
    X = df[cluster_features].copy()

    # Scale features because K-Means relies on Euclidean distance (unscaled magnitudes would dominate).
    # K-Means uses Euclidean distances; scaling avoids domination by large-scale features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate a small range of k values to keep the analysis readable in the report.
    ks = [2, 3, 4]
    k_results: List[dict] = []

    for k in ks:
        # Fit K-Means for this k and assign each listing to a cluster.
        # Fit K-Means for this k and assign each listing to a cluster.
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        # Attach cluster assignments to a copy of the engineered dataframe for summarisation.
        df_k = df.copy()
        clusters = kmeans.fit_predict(X_scaled)
        df_k["cluster"] = clusters

        # Inertia: within-cluster SSE (lower is better, mainly used for elbow diagnostics).
        inertia = float(kmeans.inertia_)
        # Silhouette is only defined for k>1; higher is better.
        sil = float(silhouette_score(X_scaled, clusters)) if k > 1 else np.nan
        k_results.append({"k": k, "inertia": inertia, "silhouette": sil})

        # Save per-cluster means/medians for key features + revenue (revenue for interpretation only).
        summary_cols = cluster_features + ["realSum"]
        cluster_summary = df_k.groupby("cluster")[summary_cols].agg(["mean", "median"])
        summary_path = plots_dir / f"cluster_summary_k{k}.csv"
        cluster_summary.to_csv(summary_path)
        print(f"Saved cluster summary (k={k}) → {summary_path}")

        # Plot mean revenue by cluster to interpret whether some clusters tend to earn more.
        mean_revenue = df_k.groupby("cluster")["realSum"].mean()
        fig, ax = plt.subplots(figsize=(5, 4))
        mean_revenue.plot(kind="bar", ax=ax)
        ax.set_title(f"Mean Revenue by Cluster (k={k})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Mean revenue (€)")
        plt.tight_layout()
        plot_path = plots_dir / f"cluster_mean_revenue_k{k}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved cluster revenue plot (k={k}) → {plot_path}")

    # Collect diagnostics across k into a single table.
    k_df = pd.DataFrame(k_results).sort_values("k")
    k_metrics_path = plots_dir / "kmeans_k_diagnostics.csv"
    k_df.to_csv(k_metrics_path, index=False)
    print(f"Saved KMeans k diagnostics → {k_metrics_path}")
    # Rank k values by silhouette for a quick model-comparison table.
    comp_path = plots_dir / "clustering_model_comparison.csv"
    sorted_k = k_df.sort_values("silhouette", ascending=False)
    sorted_k.to_csv(comp_path, index=False)
    print(f"Saved clustering model comparison table → {comp_path}")

    # Visual comparison of silhouette across k (quick diagnostic for the report appendix).
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_k.plot(x="k", y="silhouette", kind="bar", ax=ax, legend=False)
    ax.set_title("Silhouette by k")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    plt.tight_layout()
    plot_path = plots_dir / "clustering_model_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved clustering comparison plot → {plot_path}")

    # Elbow curve: inertia as a function of k.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_df["k"], k_df["inertia"], marker="o")
    ax.set_title("KMeans Elbow (Inertia)")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    plt.tight_layout()
    elbow_path = plots_dir / "kmeans_elbow_inertia.png"
    fig.savefig(elbow_path, dpi=300)
    plt.close(fig)
    print(f"Saved elbow plot → {elbow_path}")

    # Silhouette curve: silhouette score as a function of k.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_df["k"], k_df["silhouette"], marker="o")
    ax.set_title("KMeans Silhouette")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    plt.tight_layout()
    sil_path = plots_dir / "kmeans_silhouette.png"
    fig.savefig(sil_path, dpi=300)
    plt.close(fig)
    print(f"Saved silhouette plot → {sil_path}")

    return k_df
