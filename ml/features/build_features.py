from pathlib import Path

import pandas as pd


INPUT_PATH = Path("ml/data/processed/btc-usd_clean.csv")
OUTPUT_DIR = Path("ml/data/processed")


def load_data() -> pd.DataFrame:
    """Load cleaned market data."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    data["date"] = pd.to_datetime(data["date"])

    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build time-series features."""

    data = data.copy()

    # Daily return.
    data["return_1d"] = data["close"].pct_change()

    # Seven-day return.
    data["return_7d"] = data["close"].pct_change(7)

    # Moving averages.
    data["sma_7"] = data["close"].rolling(7).mean()
    data["sma_30"] = data["close"].rolling(30).mean()

    # Exponential moving averages.
    data["ema_7"] = data["close"].ewm(
        span=7,
        adjust=False,
    ).mean()

    data["ema_30"] = data["close"].ewm(
        span=30,
        adjust=False,
    ).mean()

    # Rolling volatility.
    data["volatility_7"] = (
        data["return_1d"]
        .rolling(7)
        .std()
    )

    data["volatility_30"] = (
        data["return_1d"]
        .rolling(30)
        .std()
    )

    # Volume change.
    data["volume_change"] = (
        data["volume"].pct_change()
    )

    # Daily price range.
    data["price_range"] = (
        (data["high"] - data["low"])
        / data["close"]
    )

    # Target: NEXT day's return.
    #
    # shift(-1) moves tomorrow's return
    # onto today's row.
    data["target_return_1d"] = (
        data["close"].pct_change().shift(-1)
    )

    return data


def save_features(data: pd.DataFrame) -> Path:
    """Save feature dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / "btc-usd_features.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nFeatures saved to: {output_path}"
    )

    return output_path


def main() -> None:
    print("Loading cleaned data...")

    data = load_data()

    print(f"Input rows: {len(data)}")

    print("\nBuilding features...")

    data = build_features(data)

    print(
        f"Feature dataset rows before "
        f"cleaning: {len(data)}"
    )

    # Rolling features create NaN values
    # at the beginning of the dataset.
    #
    # The final row has no target because
    # there is no "tomorrow" available.
    data = data.dropna().reset_index(drop=True)

    print(
        f"Feature dataset rows after "
        f"cleaning: {len(data)}"
    )

    print("\nFeature columns:")

    print(data.columns.tolist())

    print("\nLast 5 rows:")

    print(
        data[
            [
                "date",
                "close",
                "return_1d",
                "sma_7",
                "sma_30",
                "volatility_7",
                "target_return_1d",
            ]
        ].tail()
    )

    save_features(data)


if __name__ == "__main__":
    main()