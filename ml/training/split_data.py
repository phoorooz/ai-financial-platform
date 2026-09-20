from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "ml/data/processed/btc-usd_features.csv"
)

OUTPUT_DIR = Path("ml/data/processed")

FEATURE_COLUMNS = [
    "return_1d",
    "return_7d",
    "sma_7",
    "sma_30",
    "ema_7",
    "ema_30",
    "volatility_7",
    "volatility_30",
    "volume_change",
    "price_range",
]

TARGET_COLUMN = "target_return_1d"


def load_data() -> pd.DataFrame:
    """Load feature dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    data["date"] = pd.to_datetime(data["date"])

    return data


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically."""

    split_index = int(len(data) * 0.8)

    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    return train, test


def save_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Save train and test datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        OUTPUT_DIR / "btc-usd_train.csv"
    )

    test_path = (
        OUTPUT_DIR / "btc-usd_test.csv"
    )

    train.to_csv(
        train_path,
        index=False,
    )

    test.to_csv(
        test_path,
        index=False,
    )

    print(f"\nTrain data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")


def main() -> None:

    print("Loading feature dataset...")

    data = load_data()

    print(f"Total rows: {len(data)}")

    train, test = split_data(data)

    print("\nTime-series split:")
    print("------------------")

    print(
        f"Train rows: {len(train)}"
    )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        f"\nTrain period: "
        f"{train['date'].min().date()} "
        f"→ "
        f"{train['date'].max().date()}"
    )

    print(
        f"Test period: "
        f"{test['date'].min().date()} "
        f"→ "
        f"{test['date'].max().date()}"
    )

    print("\nFeatures:")
    print(FEATURE_COLUMNS)

    print("\nTarget:")
    print(TARGET_COLUMN)

    save_data(train, test)


if __name__ == "__main__":
    main()