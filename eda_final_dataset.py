"""
Exploratory Data Analysis for FinalDataSet.csv.
Uses pandas, numpy, seaborn, and matplotlib.
Update TARGET_COL if you want supervised target analyses; leave as None to skip.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plot style configuration to keep charts readable and consistent
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True, "figure.figsize": (8, 5)})
plt.rcParams.update({"figure.max_open_warning": 200})  # avoid noisy warnings with many plots

# --- Configuration ---
DATA_PATH = "FinalDataSet.csv"
TARGET_COL = None  # e.g., "price" or "label"; set to None if there is no target
REVENUE_COL = "realSum"  # column representing revenue


def main() -> None:
    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Quick peek at the first rows to understand schema and sample values
    print("=== Preview of data (head) ===")
    print(df.head(), "\n")

    # Separate column lists for later visualizations
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Basic information ---
    print("=== Shape (rows, columns) ===")
    print(df.shape, "\n")

    print("=== Data types ===")
    print(df.dtypes, "\n")

    print("=== Missing values per column ===")
    print(df.isna().sum().sort_values(ascending=False), "\n")

    print("=== Missing value ratio (%) per column ===")
    print((df.isna().mean().sort_values(ascending=False) * 100).round(2), "\n")

    # Simple duplicate/constant checks to flag data quality issues early
    dup_count = df.duplicated().sum()
    print(f"=== Duplicate rows ===\n{dup_count} duplicated rows\n")
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    if constant_cols:
        print("=== Constant columns (single unique value) ===")
        print(constant_cols, "\n")
    else:
        print("=== Constant columns ===\nNone\n")

    # Descriptive statistics for numeric columns
    if numeric_cols:
        print("=== Numeric summary (describe) ===")
        print(df[numeric_cols].describe().T, "\n")

    # Descriptive summary for categorical columns (unique counts and top values)
    if categorical_cols:
        print("=== Categorical summary (unique counts & top frequency) ===")

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
        print(cat_summary, "\n")

    # Quick data quality signals: missingness and skewness
    print("=== Data quality signals ===")
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    high_missing = missing_ratio[missing_ratio > 0.1]  # flag columns with >10% missing
    if not high_missing.empty:
        print("Columns with >10% missing:\n", high_missing)
    skewness = df[numeric_cols].skew().sort_values(ascending=False) if numeric_cols else pd.Series(dtype=float)
    high_skew = skewness[skewness.abs() > 1]
    if not high_skew.empty:
        print("\nSkewed numeric columns (|skew|>1):\n", high_skew)
    if high_missing.empty and high_skew.empty:
        print("No obvious missing-value or skew issues detected.")
    print()

    # Report top correlated numeric feature pairs to spot multicollinearity risks
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        # Keep upper triangle without diagonal
        tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_pairs = corr_matrix.where(tri_mask).stack().sort_values(ascending=False)
        top_corr_pairs = corr_pairs.head(10)
        if not top_corr_pairs.empty:
            print("=== Top correlated numeric feature pairs (abs corr) ===")
            print(top_corr_pairs, "\n")

    # --- Revenue vs Crime/Safety analysis ---
    if REVENUE_COL not in df.columns:
        print(f"Revenue column '{REVENUE_COL}' not found; skipping revenue analysis.\n")
    else:
        print("=== Revenue by City x DayType with Crime/Safety context ===")
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
        print(revenue_stats_sorted.to_string(index=False), "\n")

        # Boxplot: revenue distribution by City, split by DayType (top 10 cities to avoid clutter)
        top_cities = df["City"].value_counts().head(10).index
        top_subset = df[df["City"].isin(top_cities)]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=top_subset, x="City", y=REVENUE_COL, hue="DayType")
        plt.title(f"{REVENUE_COL} distribution by City (top 10) and DayType")
        plt.xlabel("City")
        plt.ylabel(REVENUE_COL)
        plt.xticks(rotation=45, ha="right")

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

    # --- Visualizations ---

    # Histograms for numeric variables to inspect distributions
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

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
        else:
            print(f"Skipping bar plot for {col} (high cardinality: {unique_vals})")

    # Correlation heatmap for numeric variables to spot linear relationships
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (numeric features)")

    # Boxplots for outlier detection on top-variance numeric columns
    if numeric_cols:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        top_box_cols = variances.index.tolist()[: min(len(variances), 8)]  # plot up to 8
        for col in top_box_cols:
            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot: {col}")
            plt.xlabel(col)

    # --- Supervised learning relationships (only if a target column is defined) ---
    if TARGET_COL and TARGET_COL in df.columns:
        print(f"=== Supervised analysis vs target: {TARGET_COL} ===\n")

        # Correlation of numeric features with numeric target to assess strength/direction
        if TARGET_COL in numeric_cols:
            corr_to_target = df[numeric_cols].corr()[TARGET_COL].drop(TARGET_COL)
            print("Correlation of numeric features with target:\n", corr_to_target.sort_values(ascending=False), "\n")

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

        # Boxplots of numeric features grouped by categorical target for separation insight
        if TARGET_COL in categorical_cols:
            for col in numeric_cols:
                plt.figure()
                sns.boxplot(x=df[TARGET_COL], y=df[col])
                plt.title(f"{col} by {TARGET_COL}")
                plt.xlabel(TARGET_COL)
                plt.ylabel(col)

        # Count plots for categorical features grouped by target to view class balance shifts
        if TARGET_COL in categorical_cols:
            for col in categorical_cols:
                if col == TARGET_COL:
                    continue
                plt.figure()
                sns.countplot(data=df, x=col, hue=TARGET_COL)
                plt.title(f"{col} distribution by {TARGET_COL}")
                plt.xticks(rotation=45, ha="right")
    else:
        print("No supervised target set; set TARGET_COL to enable target-based analyses.\n")

    # Display all generated figures
    plt.show()


if __name__ == "__main__":
    main()
