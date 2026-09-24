from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor


TEST_PATH = Path(
    "ml/data/processed/btc-usd_test.csv"
)

MODEL_PATH = Path(
    "ml/models/xgboost_btc_return.json"
)

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


def load_test_data() -> pd.DataFrame:
    """Load test dataset."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    return pd.read_csv(TEST_PATH)


def load_model() -> XGBRegressor:
    """Load trained XGBoost model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    return model


def calculate_directional_accuracy(
    actual: pd.Series,
    predictions: pd.Series,
) -> float:
    """Calculate percentage of correct directions."""

    actual_direction = actual >= 0
    predicted_direction = predictions >= 0

    return (
        (actual_direction == predicted_direction)
        .mean()
    )


def evaluate(
    model: XGBRegressor,
    data: pd.DataFrame,
) -> dict:
    """Evaluate model on unseen test data."""

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    mse = mean_squared_error(
        y,
        predictions,
    )

    rmse = mse ** 0.5

    directional_accuracy = (
        calculate_directional_accuracy(
            y,
            predictions,
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }


def main() -> None:
    print("Loading test data...")

    data = load_test_data()

    print(
        f"Test observations: {len(data)}"
    )

    print("\nLoading XGBoost model...")

    model = load_model()

    print("Model loaded successfully.")

    mlflow.set_experiment(
        "bitcoin_return_prediction"
    )

    print("\nStarting MLflow run...")

    with mlflow.start_run() as run:

        # Record which local model file was used
        # for this evaluation.
        mlflow.log_param(
            "model_file",
            str(MODEL_PATH),
        )

        metrics = evaluate(
            model,
            data,
        )

        mlflow.log_metrics(
            metrics
        )

        print("\nXGBoost Evaluation")
        print("------------------")

        print(
            f"MAE:  "
            f"{metrics['mae']:.6f}"
        )

        print(
            f"RMSE: "
            f"{metrics['rmse']:.6f}"
        )

        print(
            f"Directional Accuracy: "
            f"{metrics['directional_accuracy']:.2%}"
        )

        print(
            f"\nMLflow Run ID: "
            f"{run.info.run_id}"
        )


if __name__ == "__main__":
    main()