"""
Create a log-transformed dataset from FinalDataSet_geo_merged.csv.

Overwrites `realSum` with log1p(realSum) to reduce skew; no extra column is added.
Non-positive or non-numeric realSum values are coerced to NaN before log1p.
Outputs FinalDataSet_geo_merged_log.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH = Path("FinalDataSet_geo_merged.csv")
OUTPUT_PATH = Path("FinalDataSet_geo_merged_log.csv")

TARGET_COL = "realSum"


def main() -> None:
    # --- Input validation ---
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if TARGET_COL not in df.columns:
        raise SystemExit(
            f"Column '{TARGET_COL}' not found in input file: {INPUT_PATH}"
        )

    # --- Cleaning ---
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Mark invalid values
    invalid_mask = df[TARGET_COL] <= 0
    invalid_count = int(invalid_mask.sum())
    df.loc[invalid_mask, TARGET_COL] = np.nan

    # --- Log transform (overwrite realSum) ---
    df[TARGET_COL] = np.log1p(df[TARGET_COL])

    # --- Output ---
    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(df):,} rows to {OUTPUT_PATH}. "
        f"Invalid or non-positive {TARGET_COL} values converted to NaN: {invalid_count}"
    )


if __name__ == "__main__":
    main()
