from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


TEST_PATH = Path(
    "ml/data/processed/btc-usd_test.csv"
)


def load_test_data() -> pd.DataFrame:
    """Load test dataset."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    data = pd.read_csv(TEST_PATH)

    return data


def evaluate_baseline(data: pd.DataFrame) -> None:
    """Evaluate naive forecasting baseline."""

    # Today's return as tomorrow's prediction.
    predictions = data["return_1d"]

    actual = data["target_return_1d"]

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    mse = mean_squared_error(
        actual,
        predictions,
    )

    rmse = mse ** 0.5

    print("\nBaseline Results")
    print("----------------")

    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")


def main() -> None:

    print("Loading test data...")

    data = load_test_data()

    print(
        f"Test observations: {len(data)}"
    )

    evaluate_baseline(data)


if __name__ == "__main__":
    main()