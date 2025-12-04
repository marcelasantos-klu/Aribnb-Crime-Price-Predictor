"""
Structured explanation of the original datasets before merging.
Outputs counts of files/columns/rows and common preprocessing challenges.
All output is written to overview_output.txt for easy sharing.
"""
from pathlib import Path
import io
import pandas as pd

OUTPUT_TXT = "overview_output.txt"
RAW_DIR = Path("Raw Data")
CRIME_PATH = Path("Indices/World Crime Index .csv")


def main() -> None:
    # Capture log lines in-memory so we can write once to disk at the end; avoids partial files.
    buffer = io.StringIO()

    def log(line: str = "") -> None:
        print(line)
        print(line, file=buffer)

    def log_kv(key: str, value) -> None:
        log(f"- {key}: {value}")

    # --- Airbnb dataset overview (Raw Data folder) ---
    airbnb_files = sorted(RAW_DIR.glob("*.csv"))
    log("=== Source datasets (pre-merge) overview ===")
    log()
    log("1) Airbnb dataset (Raw Data folder)")
    if not airbnb_files:
        log("- No Airbnb CSVs found in Raw Data.")
    else:
        # Use the first file to inspect schema (columns), but aggregate row counts across all city/day files.
        sample_df = pd.read_csv(airbnb_files[0])
        total_rows = 0
        for f in airbnb_files:
            total_rows += pd.read_csv(f).shape[0]
        cities = {f.stem.split("_", 1)[0] for f in airbnb_files}
        daytypes = {f.stem.split("_", 1)[1] for f in airbnb_files if "_" in f.stem}

        log_kv("Number of city-daytype files", len(airbnb_files))
        log_kv("Total rows across files", total_rows)
        log_kv("Columns (count)", f"{len(sample_df.columns)} columns: {list(sample_df.columns)}")
        log_kv("Distinct cities from filenames", sorted(cities))
        log_kv("Distinct day types from filenames", sorted(daytypes))
        log("- City and DayType are not stored as columns; they are encoded in filenames (e.g., athens_weekdays).")
        log("- Main features include pricing (realSum), room/bedrooms/capacity, host/superhost flags,")
        log("  cleanliness and guest satisfaction ratings, distance/attractiveness indices, and geo-coordinates.")

    log()
    log("2) Crime dataset (Indices folder)")
    if not CRIME_PATH.exists():
        log("- Crime dataset not found at expected path.")
    else:
        crime_df = pd.read_csv(CRIME_PATH)
        log_kv("Rows", crime_df.shape[0])
        log_kv("Columns (count)", f"{crime_df.shape[1]} columns: {list(crime_df.columns)}")
        log("- Relevant columns: City, Crime Index, Safety Index (Rank column is irrelevant).")
        log("- City values combine city and country in one string (e.g., 'London, United Kingdom').")
        dup_cities = crime_df["City"].value_counts()
        duplicate_labels = dup_cities[dup_cities > 1]
        if not duplicate_labels.empty:
            log_kv("Duplicate city labels", duplicate_labels.to_dict())
        else:
            log_kv("Duplicate city labels", "None")
    log()
    log("3) Preprocessing challenges before merging")
    log("- Different storage of city name: Airbnb uses filenames; crime data uses 'City, Country' strings.")
    log("- Case/whitespace differences require normalizing to lowercase and stripping spaces.")
    log("- Country suffix in crime data must be split off to align with Airbnb city names.")
    log("- Duplicate city label (London) needs disambiguation to keep the intended country.")
    log("- Building a consistent merge key (cleaned city name) is required for a reliable join.")
    log()
    log("4) Why extract and clean city names for merging")
    log("- Without extracting city from filenames and stripping country from crime data, keys would not match.")
    log("- Normalizing case/whitespace prevents missed matches due to formatting.")
    log("- Removing duplicate/ambiguous entries (e.g., London, Canada) avoids incorrect joins.")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())


if __name__ == "__main__":
    main()
