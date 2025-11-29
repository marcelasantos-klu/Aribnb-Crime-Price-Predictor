from pathlib import Path

import pandas as pd

AIRBNB_PATH = Path("FinalAirbnb.csv")
CRIME_PATH = Path("Indices/World Crime Index .csv")
OUTPUT_PATH = Path("FinalDataSet.csv")


def get_city_column(df: pd.DataFrame) -> str:
    if "city" in df.columns:
        return "city"
    if "City" in df.columns:
        return "City"
    raise KeyError("No 'city' or 'City' column found in FinalAirbnb.csv")


def main() -> None:
    airbnb = pd.read_csv(AIRBNB_PATH)
    airbnb = airbnb.loc[:, ~airbnb.columns.str.startswith("Unnamed:")]

    city_col = get_city_column(airbnb)
    airbnb["_merge_key"] = airbnb[city_col].str.lower().str.strip()

    crime = pd.read_csv(CRIME_PATH, usecols=["City", "Crime Index", "Safety Index"])
    crime = crime.assign(City=crime["City"].str.split(",", n=1).str[0].str.strip())
    crime["_merge_key"] = crime["City"].str.lower().str.strip()

    merged = airbnb.merge(
        crime[["_merge_key", "Crime Index", "Safety Index"]],
        on="_merge_key",
        how="left",
    ).drop(columns="_merge_key")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote merged dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
