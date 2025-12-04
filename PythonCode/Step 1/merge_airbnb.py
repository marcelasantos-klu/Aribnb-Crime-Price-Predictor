import csv
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("Raw Data")  # folder containing the source CSV files
OUTPUT_FILE = Path("FinalAirbnb.csv")
DEFAULT_DELIMITER = ","  # fallback delimiter if sniffing fails


def detect_delimiter(path: Path, fallback: str = DEFAULT_DELIMITER) -> str:
    """
    Return the detected delimiter (',' or ';') for the given CSV file.
    Some source files use semicolons; sniffing avoids hard-coding a separator that could fail.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";"]).delimiter
    except csv.Error:
        return fallback


def main() -> None:
    """Combine city/day-type Airbnb CSVs into one normalized dataset."""
    frames = []

    # Iterate predictable file names like "amsterdam_weekdays.csv" sorted for deterministic output ordering.
    for csv_path in sorted(RAW_DATA_DIR.glob("*_*.csv")):
        city, day_type = csv_path.stem.split("_", 1)
        sep = detect_delimiter(csv_path)
        # Preserve per-file delimiter to avoid mis-parsing when sources mix ',' and ';'.
        df = pd.read_csv(csv_path, sep=sep)
        # Inject city/day type as explicit columns so downstream merges don't rely on filenames.
        df["City"] = city
        df["DayType"] = day_type
        frames.append(df)

    if not frames:
        raise SystemExit("No matching CSV files found in Raw Data")

    final_df = pd.concat(frames, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(final_df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
