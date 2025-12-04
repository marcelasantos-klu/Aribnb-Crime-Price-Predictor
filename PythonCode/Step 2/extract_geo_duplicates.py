"""
Extract rows that share the same lng/lat pair from FinalDataSet_geo_merged.csv.
Outputs only rows whose lng/lat occur more than once into FinalDataSet_geo_duplicates.csv,
sorted so duplicates sit together, with helper columns to inspect differences.
"""
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("FinalDataSet_geo_merged.csv")
OUTPUT_PATH = Path("FinalDataSet_geo_duplicates.csv")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    # Flag rows sharing identical lng/lat regardless of position to surface location collisions.
    dup_mask = df.duplicated(subset=["lng", "lat"], keep=False)
    dup_df = df.loc[dup_mask].copy()

    # Add a count column showing how many times each pair appears
    counts = df.groupby(["lng", "lat"]).size()
    dup_df["lng_lat_count"] = dup_df.set_index(["lng", "lat"]).index.map(counts)

    # Order rows so duplicates sit together; add within-group order to compare differences
    dup_df = dup_df.sort_values(["lng_lat_count", "lng", "lat"], ascending=[False, True, True]).copy()
    dup_df["dup_rank_within_pair"] = dup_df.groupby(["lng", "lat"]).cumcount() + 1

    dup_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dup_df)} rows with duplicated lng/lat to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
