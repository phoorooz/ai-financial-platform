from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


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


class BitcoinReturnPredictor:
    """Load and run the trained XGBoost model."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = XGBRegressor()

        self.model.load_model(model_path)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> float:
        """Predict the next-day return."""

        missing_features = (
            set(FEATURE_COLUMNS)
            - set(features.columns)
        )

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )

        X = features[FEATURE_COLUMNS]

        prediction = self.model.predict(X)

        return float(prediction[0])


def main() -> None:

    print("Loading model...")

    predictor = BitcoinReturnPredictor()

    print("Model loaded successfully.")

    # Load the latest feature row.
    features_path = Path(
        "ml/data/processed/btc-usd_features.csv"
    )

    data = pd.read_csv(features_path)

    latest_features = data.tail(1)

    prediction = predictor.predict(
        latest_features
    )

    print(
        f"\nPredicted next-day return: "
        f"{prediction:.4%}"
    )


if __name__ == "__main__":
    main()