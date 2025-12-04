"""
Create a log-transformed revenue column from FinalDataSet_geo_merged.csv.
Adds `realSum_log = log1p(realSum)` to improve skew handling in models, and writes
the augmented dataset to FinalDataSet_geo_merged_log.csv.
Non-positive or non-numeric realSum values are coerced to NaN before log1p.
"""
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("FinalDataSet_geo_merged.csv")
OUTPUT_PATH = Path("FinalDataSet_geo_merged_log.csv")
TARGET_COL = "realSum"
LOG_COL = f"{TARGET_COL}_log"


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Column '{TARGET_COL}' not found in {INPUT_PATH}")

    # Ensure numeric and null out invalid entries before log-transforming.
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    non_positive_mask = df[TARGET_COL] <= 0
    non_positive_count = int(non_positive_mask.sum())
    if non_positive_count:
        df.loc[non_positive_mask, TARGET_COL] = np.nan

    df[LOG_COL] = np.log1p(df[TARGET_COL])

    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Wrote {len(df)} rows to {OUTPUT_PATH}. "
        f"Non-positive or invalid {TARGET_COL} values set to NaN: {non_positive_count}"
    )


if __name__ == "__main__":
    main()
