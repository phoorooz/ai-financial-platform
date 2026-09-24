from pathlib import Path

import mlflow
import pandas as pd
from xgboost import XGBRegressor


TRAIN_PATH = Path(
    "ml/data/processed/btc-usd_train.csv"
)

MODEL_DIR = Path(
    "ml/models"
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


def load_training_data() -> pd.DataFrame:
    """Load training dataset."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    return pd.read_csv(TRAIN_PATH)


def train_model(
    data: pd.DataFrame,
) -> XGBRegressor:
    """Train XGBoost regression model."""

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    model_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    mlflow.log_params(model_params)
    mlflow.log_param(
        "training_rows",
        len(data),
    )

    model = XGBRegressor(
        **model_params
    )

    model.fit(X, y)

    return model


def save_model(
    model: XGBRegressor,
) -> Path:
    """Save trained model."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / "xgboost_btc_return.json"
    )

    model.save_model(model_path)

    print(
        f"\nModel saved to: {model_path}"
    )

    return model_path


def main() -> None:
    print("Loading training data...")

    data = load_training_data()

    print(
        f"Training observations: "
        f"{len(data)}"
    )

    print("\nFeatures:")
    print(FEATURE_COLUMNS)

    print(
        f"\nTarget: "
        f"{TARGET_COLUMN}"
    )

    mlflow.set_experiment(
        "bitcoin_return_prediction"
    )

    print(
        "\nStarting MLflow run..."
    )

    with mlflow.start_run() as run:

        print("Training XGBoost...")

        model = train_model(data)

        mlflow.xgboost.log_model(
        model,
        artifact_path="xgboost_model",
        )

        print("Training completed!")

        save_model(model)

        print(
            f"MLflow Run ID: "
            f"{run.info.run_id}"
        )

if __name__ == "__main__":
    main()
