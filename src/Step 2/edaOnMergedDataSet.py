"""
Exploratory Data Analysis for FinalDataSet.csv.
Uses pandas, numpy, seaborn, and matplotlib.
All console output is captured to OUTPUT_TXT so you have a saved log of the run.
"""
import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Plot style configuration to keep charts readable and consistent
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True, "figure.figsize": (8, 5)})
plt.rcParams.update({"figure.max_open_warning": 200})  # avoid noisy warnings with many plots

# --- Configuration ---
# Try both root and FinalDataSetEDA/ paths to be robust to the recent file move.
DATA_CANDIDATES = [
    Path("data/dataSetEDA/edaDataSet.csv"),
    Path("data/dataSetEDA/edaDataSet.csv"),
]
DATA_PATH = None  # resolved at runtime
TARGET_COL = None  # e.g., "price" or "label"; set to None if there is no target
REVENUE_COL = "realSum"  # column representing revenue
OUTPUT_TXT = "src/Step 2/eda_terminal_output.txt"  # capture all printed output here
PLOTS_DIR = Path("plots&models/EDA")
SHOW_PLOTS = False  # keep headless to avoid blocking on many plot windows


def main() -> None:
    # Resolve dataset path
    global DATA_PATH
    DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if DATA_PATH is None:
        raise SystemExit(f"No input dataset found. Checked: {DATA_CANDIDATES}")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Buffer stdout so the same text goes both to console and to file at the end.
    buffer = io.StringIO()

    def log(*args, **kwargs) -> None:
        """Print to console and capture in buffer for writing to disk."""
        print(*args, **kwargs)
        print(*args, **kwargs, file=buffer)

    def save_fig(filename: str, dpi: int = 300) -> None:
        """Save current matplotlib figure to PLOTS_DIR and log the path."""
        path = PLOTS_DIR / filename
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        log(f"Saved figure → {path}")

    try:
        # Load the dataset
        df = pd.read_csv(DATA_PATH)

        # Quick peek at the first rows to understand schema and sample values
        log("=== Preview of data (head) ===")
        log(df.head(), "\n")

        # Separate column lists for later visualizations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # --- Basic information ---
        log("=== Shape (rows, columns) ===")
        log(df.shape, "\n")

        log("=== Data types ===")
        log(df.dtypes, "\n")

        log("=== Missing values per column ===")
        log(df.isna().sum().sort_values(ascending=False), "\n")

        log("=== Missing value ratio (%) per column ===")
        log((df.isna().mean().sort_values(ascending=False) * 100).round(2), "\n")

        # Simple duplicate/constant checks to flag data quality issues early
        dup_count = df.duplicated().sum()
        log(f"=== Duplicate rows ===\n{dup_count} duplicated rows\n")
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
        if constant_cols:
            log("=== Constant columns (single unique value) ===")
            log(constant_cols, "\n")
        else:
            log("=== Constant columns ===\nNone\n")

        # Descriptive statistics for numeric columns
        if numeric_cols:
            log("=== Numeric summary (describe) ===")
            log(df[numeric_cols].describe().T, "\n")

        # Descriptive summary for categorical columns (unique counts and top values)
        if categorical_cols:
            log("=== Categorical summary (unique counts & top frequency) ===")

            def summarize_cat(s: pd.Series) -> pd.Series:
                top = s.value_counts(dropna=False).head(3)
                top_fmt = "; ".join([f"{idx}: {cnt}" for idx, cnt in top.items()])
                return pd.Series(
                    {
                        "n_unique": s.nunique(dropna=False),
                        "top_3_values": top_fmt,
                    }
                )

            cat_summary = df[categorical_cols].apply(summarize_cat)
            log(cat_summary, "\n")

        # Boolean value distribution: how often True/False per boolean column
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            log("=== Boolean columns value counts (True/False) ===")
            for col in bool_cols:
                counts = df[col].value_counts(dropna=False)
                log(f"{col}: {counts.to_dict()}")
            log()

        # room_type distribution to see category frequencies
        if "room_type" in df.columns:
            log("=== room_type value counts ===")
            log(df["room_type"].value_counts(dropna=False))
            log()

        # Bookings per city (counts) + PNG export for the report
        if "City" in df.columns:
            city_counts = df["City"].value_counts(dropna=False).sort_values(ascending=False)
            log("=== Bookings per City ===")
            log(city_counts, "\n")

            top_n = min(len(city_counts), 20)
            plt.figure(figsize=(10, 6))
            top_counts = city_counts.head(top_n)
            ax = sns.barplot(x=top_counts.index, y=top_counts.values)
            plt.title(f"Bookings per City (top {top_n})")
            plt.xlabel("City")
            plt.ylabel("Bookings")
            plt.xticks(rotation=45, ha="right")
            # Add count labels on bars
            for i, v in enumerate(top_counts.values):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
            save_fig("bookings_per_city.png")
            plt.close()
            log()
        else:
            log("City column missing; skipping bookings-per-city plot.\n")

        # Value counts for selected discrete columns
        discrete_cols = ["bedrooms", "person_capacity", "biz", "multi"]
        present_discrete = [c for c in discrete_cols if c in df.columns]
        if present_discrete:
            log("=== Value counts for selected discrete columns ===")
            for col in present_discrete:
                counts = df[col].value_counts(dropna=False).sort_index()
                log(f"{col}:\n{counts}")
            log()

        # Value counts for key rating columns to see score distributions
        rating_cols = ["cleanliness_rating", "guest_satisfaction_overall"]
        rating_present = [c for c in rating_cols if c in df.columns]
        if rating_present:
            log("=== Value counts for rating columns ===")
            for col in rating_present:
                counts = df[col].value_counts(dropna=False).sort_index()
                log(f"{col}:\n{counts}")
            log()

        # Quick data quality signals: missingness and skewness
        log("=== Data quality signals ===")
        missing_ratio = df.isna().mean().sort_values(ascending=False)
        high_missing = missing_ratio[missing_ratio > 0.1]  # flag columns with >10% missing
        if not high_missing.empty:
            log("Columns with >10% missing:\n", high_missing)
        skewness = df[numeric_cols].skew().sort_values(ascending=False) if numeric_cols else pd.Series(dtype=float)
        high_skew = skewness[skewness.abs() > 1]
        if not high_skew.empty:
            log("\nSkewed numeric columns (|skew|>1):\n", high_skew)
        if high_missing.empty and high_skew.empty:
            log("No obvious missing-value or skew issues detected.")
        log()

        # Report top correlated numeric feature pairs to spot multicollinearity risks
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().abs()
            # Keep upper triangle without diagonal to avoid duplicate/repeated pairs.
            tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_pairs = corr_matrix.where(tri_mask).stack().sort_values(ascending=False)
            top_corr_pairs = corr_pairs.head(10)
            if not top_corr_pairs.empty:
                log("=== Top correlated numeric feature pairs (abs corr) ===")
                log(top_corr_pairs, "\n")

        # --- Revenue vs Crime/Safety analysis ---
        if REVENUE_COL not in df.columns:
            log(f"Revenue column '{REVENUE_COL}' not found; skipping revenue analysis.\n")
        else:
            log("=== Revenue by City x DayType with Crime/Safety context ===")
            # Aggregate revenue stats per City-DayType to capture central tendency and spread
            revenue_stats = (
                df.groupby(["City", "DayType"])
                .agg(
                    mean_revenue=(REVENUE_COL, "mean"),
                    median_revenue=(REVENUE_COL, "median"),
                    min_revenue=(REVENUE_COL, "min"),
                    max_revenue=(REVENUE_COL, "max"),
                    std_revenue=(REVENUE_COL, "std"),
                    mean_crime_index=("Crime Index", "mean"),
                    mean_safety_index=("Safety Index", "mean"),
                    count=("City", "size"),
                )
                .reset_index()
            )

            # Sort by mean revenue (highest first) for readability
            revenue_stats_sorted = revenue_stats.sort_values(by="mean_revenue", ascending=False)
            log(revenue_stats_sorted.to_string(index=False), "\n")

            # Boxplot: revenue distribution by City, split by DayType (top 10 cities to avoid clutter)
            top_cities = df["City"].value_counts().head(10).index
            top_subset = df[df["City"].isin(top_cities)]
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=top_subset, x="City", y=REVENUE_COL, hue="DayType")
            plt.title(f"{REVENUE_COL} distribution by City (top 10) and DayType")
            plt.xlabel("City")
            plt.ylabel(REVENUE_COL)
            plt.xticks(rotation=45, ha="right")
            save_fig("revenue_by_city_daytype_box.png")
            plt.close()

            # Scatterplots to relate crime/safety metrics with revenue, colored by DayType
            plt.figure()
            sns.scatterplot(
                data=df,
                x="Crime Index",
                y=REVENUE_COL,
                hue="DayType",
                alpha=0.3,
            )
            plt.title(f"{REVENUE_COL} vs Crime Index by DayType")
            plt.xlabel("Crime Index")
            plt.ylabel(REVENUE_COL)
            save_fig("revenue_vs_crime_index.png")
            plt.close()

            plt.figure()
            sns.scatterplot(
                data=df,
                x="Safety Index",
                y=REVENUE_COL,
                hue="DayType",
                alpha=0.3,
            )
            plt.title(f"{REVENUE_COL} vs Safety Index by DayType")
            plt.xlabel("Safety Index")
            plt.ylabel(REVENUE_COL)
            save_fig("revenue_vs_safety_index.png")
            plt.close()

        # --- Visualizations ---

        # Histograms for numeric variables to inspect distributions
        for col in numeric_cols:
            plt.figure()
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f"Histogram: {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            save_fig(f"hist_{col}.png")
            plt.close()

        # Bar plots for categorical variables (skip if too many categories to avoid clutter)
        for col in categorical_cols:
            unique_vals = df[col].nunique(dropna=False)
            if unique_vals <= 20:
                plt.figure()
                counts = df[col].value_counts(dropna=False)
                sns.barplot(x=counts.index, y=counts.values)
                plt.title(f"Bar Plot: {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                save_fig(f"bar_{col}.png")
                plt.close()
            else:
                log(f"Skipping bar plot for {col} (high cardinality: {unique_vals})")

        # Correlation heatmap for numeric variables to spot linear relationships
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap (numeric features)")
            save_fig("correlation_heatmap.png")
            plt.close()

        # Boxplots for outlier detection on top-variance numeric columns
        if numeric_cols:
            variances = df[numeric_cols].var().sort_values(ascending=False)
            # Focus on highest-variance columns where outliers are more informative; cap plot count for readability.
            top_box_cols = variances.index.tolist()[: min(len(variances), 8)]
            for col in top_box_cols:
                plt.figure()
                sns.boxplot(x=df[col])
                plt.title(f"Boxplot: {col}")
                plt.xlabel(col)
                save_fig(f"boxplot_{col}.png")
                plt.close()

        # --- Supervised learning relationships (only if a target column is defined) ---
        if TARGET_COL and TARGET_COL in df.columns:
            log(f"=== Supervised analysis vs target: {TARGET_COL} ===\n")

            # Correlation of numeric features with numeric target to assess strength/direction
            if TARGET_COL in numeric_cols:
                corr_to_target = df[numeric_cols].corr()[TARGET_COL].drop(TARGET_COL)
                log("Correlation of numeric features with target:\n", corr_to_target.sort_values(ascending=False), "\n")

            # Scatterplots for numeric features against numeric target to inspect patterns
            if TARGET_COL in numeric_cols:
                for col in numeric_cols:
                    if col == TARGET_COL:
                        continue
                    plt.figure()
                    sns.scatterplot(x=df[col], y=df[TARGET_COL])
                    plt.title(f"{col} vs {TARGET_COL}")
                    plt.xlabel(col)
                    plt.ylabel(TARGET_COL)
                    save_fig(f"scatter_{col}_vs_{TARGET_COL}.png")
                    plt.close()

            # Boxplots of numeric features grouped by categorical target for separation insight
            if TARGET_COL in categorical_cols:
                for col in numeric_cols:
                    plt.figure()
                    sns.boxplot(x=df[TARGET_COL], y=df[col])
                    plt.title(f"{col} by {TARGET_COL}")
                    plt.xlabel(TARGET_COL)
                    plt.ylabel(col)
                    save_fig(f"box_{col}_by_{TARGET_COL}.png")
                    plt.close()

            # Count plots for categorical features grouped by target to view class balance shifts
            if TARGET_COL in categorical_cols:
                for col in categorical_cols:
                    if col == TARGET_COL:
                        continue
                    plt.figure()
                    sns.countplot(data=df, x=col, hue=TARGET_COL)
                    plt.title(f"{col} distribution by {TARGET_COL}")
                    plt.xticks(rotation=45, ha="right")
                    save_fig(f"count_{col}_by_{TARGET_COL}.png")
                    plt.close()
        else:
            log("No supervised target set; set TARGET_COL to enable target-based analyses.\n")

        # Display or close all generated figures
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close("all")

    finally:
        # Persist full log to disk for later reference
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())


if __name__ == "__main__":
    main()
