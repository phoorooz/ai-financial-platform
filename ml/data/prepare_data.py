from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("ml/data/raw/btc-usd_historical.csv")
PROCESSED_DATA_DIR = Path("ml/data/processed")


REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def load_data() -> pd.DataFrame:
    """Load raw market data."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def validate_data(data: pd.DataFrame) -> None:
    """Validate the structure and values of market data."""

    print("Validating data...")

    # Check required columns.
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Convert date.
    data["date"] = pd.to_datetime(data["date"])

    # Check chronological order.
    if not data["date"].is_monotonic_increasing:
        raise ValueError("Dates are not sorted in ascending order.")

    # Check duplicate dates.
    duplicate_dates = data["date"].duplicated().sum()

    if duplicate_dates > 0:
        raise ValueError(
            f"Found {duplicate_dates} duplicate dates."
        )

    # Check missing values.
    missing_values = data[REQUIRED_COLUMNS].isna().sum()

    if missing_values.any():
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
        raise ValueError("Missing values detected.")

    # OHLC validation.
    invalid_high_low = (data["high"] < data["low"]).sum()

    if invalid_high_low > 0:
        raise ValueError(
            f"Found {invalid_high_low} rows where high < low."
        )

    invalid_open = (
        (data["open"] > data["high"])
        | (data["open"] < data["low"])
    ).sum()

    if invalid_open > 0:
        raise ValueError(
            f"Found {invalid_open} invalid open prices."
        )

    invalid_close = (
        (data["close"] > data["high"])
        | (data["close"] < data["low"])
    ).sum()

    if invalid_close > 0:
        raise ValueError(
            f"Found {invalid_close} invalid close prices."
        )

    # Volume validation.
    if (data["volume"] < 0).any():
        raise ValueError("Negative volume detected.")

    print("Validation passed!")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize market data."""

    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])

    # Ensure chronological order.
    data = data.sort_values("date")

    # Remove duplicate dates if they exist.
    data = data.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    # Reset index.
    data = data.reset_index(drop=True)

    return data


def save_data(data: pd.DataFrame) -> Path:
    """Save processed data."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "btc-usd_clean.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print(f"\nClean data saved to: {output_path}")

    return output_path


def main() -> None:
    data = load_data()

    print(f"Loaded {len(data)} rows.")

    validate_data(data)

    data = clean_data(data)

    print(f"Cleaned data contains {len(data)} rows.")

    save_data(data)


if __name__ == "__main__":
    main()