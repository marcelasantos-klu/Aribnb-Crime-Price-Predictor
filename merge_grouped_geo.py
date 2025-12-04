"""
Merge Airbnb listings by shared geo/room attributes, averaging selected metrics.
Grouping: room_type, room_shared, room_private, person_capacity, multi, bedrooms, lng, lat, City.
Removes columns: DayType, room_shared, room_private, rest_index, attr_index, multi in the output.
Adds geo_id as "{lng}_{lat}".
Outputs one row per unique combination of the grouping attributes.
"""
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("FinalDataSet.csv")
OUTPUT_PATH = Path("FinalDataSet_geo_merged.csv")

GROUP_COLS = [
    "room_type",
    "room_shared",
    "room_private",
    "person_capacity",
    "multi",
    "bedrooms",
    "lng",
    "lat",
    "City",
]

MEAN_COLS = [
    "realSum",
    "metro_dist",
    "attr_index_norm",
    "rest_index_norm",
]

DROP_AFTER = ["DayType", "room_shared", "room_private", "rest_index", "attr_index", "multi"]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Convert boolean columns to binary ints (True->1, False->0) for consistency
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # Ensure biz is binary 0/1 even if typed as bool
    if "biz" in df.columns:
        if df["biz"].dtype == "bool":
            df["biz"] = df["biz"].astype(int)

    missing_group = [c for c in GROUP_COLS if c not in df.columns]
    missing_mean = [c for c in MEAN_COLS if c not in df.columns]
    if missing_group or missing_mean:
        raise SystemExit(
            f"Missing required columns. Group missing: {missing_group}; Mean missing: {missing_mean}"
        )

    # Build aggregation: take means for numeric metrics; keep first value for non-group metadata columns.
    # Using "first" preserves representative attributes without inflating row counts.
    other_cols = [c for c in df.columns if c not in GROUP_COLS + MEAN_COLS]
    agg = {col: "mean" for col in MEAN_COLS}
    for col in other_cols:
        agg[col] = "first"

    merged = (
        df.groupby(GROUP_COLS, dropna=False)
        .agg(agg)
        .reset_index()
    )

    # Drop unwanted columns after grouping
    merged = merged.drop(columns=[c for c in DROP_AFTER if c in merged.columns], errors="ignore")

    # Add geo_id to preserve location identity while dropping separate lng/lat columns
    merged["geo_id"] = merged.apply(lambda r: f"{r['lng']}_{r['lat']}", axis=1)

    # Drop lng/lat now that geo_id is present
    merged = merged.drop(columns=["lng", "lat"], errors="ignore")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote merged dataset to {OUTPUT_PATH} with {len(merged)} rows.")


if __name__ == "__main__":
    main()
