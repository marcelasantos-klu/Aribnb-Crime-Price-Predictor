from pathlib import Path

import pandas as pd

AIRBNB_PATH = Path("FinalAirbnb.csv")
CRIME_PATH = Path("Indices/World Crime Index .csv")
OUTPUT_PATH = Path("FinalDataSet.csv")


def get_city_column(df: pd.DataFrame) -> str:
    """Return the column name to use for city matching, handling capitalization."""
    if "city" in df.columns:
        return "city"
    if "City" in df.columns:
        return "City"
    raise KeyError("No 'city' or 'City' column found in FinalAirbnb.csv")


def main() -> None:
    """Merge Airbnb listings with crime/safety indices by city."""
    # Load Airbnb export and drop unnamed index columns that appear after CSV saves
    airbnb = pd.read_csv(AIRBNB_PATH)
    airbnb = airbnb.loc[:, ~airbnb.columns.str.startswith("Unnamed:")]

    city_col = get_city_column(airbnb)
    # Normalize merge key to be case-insensitive and whitespace-tolerant
    airbnb["_merge_key"] = airbnb[city_col].str.lower().str.strip()

    # Load crime data with only the relevant score columns
    crime = pd.read_csv(CRIME_PATH, usecols=["City", "Crime Index", "Safety Index"])

    # Split "City, Country" and drop the Canada entry for London to disambiguate
    city_country = crime["City"].str.split(",", n=1, expand=True)
    crime["City_only"] = city_country[0].str.strip()
    crime["Country"] = city_country[1].str.strip()

    # Keep London, United Kingdom and drop London, Canada to avoid double matches
    crime = crime[
        (crime["City_only"].str.lower() != "london")
        | (
            (crime["City_only"].str.lower() == "london")
            & (crime["Country"] == "United Kingdom")
        )
    ]

    # Reset City to the clean city name and discard helper columns
    crime["City"] = crime["City_only"]
    crime = crime.drop(columns=["City_only", "Country"])

    # Build the same normalized merge key as for the Airbnb frame
    crime["_merge_key"] = crime["City"].str.lower().str.strip()

    # Perform a left merge so all Airbnb rows are preserved
    merged = airbnb.merge(
        crime[["_merge_key", "Crime Index", "Safety Index"]],
        on="_merge_key",
        how="left",
    ).drop(columns="_merge_key")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote merged dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
