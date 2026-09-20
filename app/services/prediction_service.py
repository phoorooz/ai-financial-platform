from pathlib import Path

import pandas as pd

from ml.inference.predict import BitcoinReturnPredictor


FEATURES_PATH = Path(
    "ml/data/processed/btc-usd_features.csv"
)


class PredictionService:
    """Business logic for Bitcoin predictions."""

    def __init__(self):
        self.predictor = BitcoinReturnPredictor()

    def predict(
        self,
        features: dict,
    ) -> float:
        """Predict the next-day return from features."""

        feature_dataframe = pd.DataFrame(
            [features]
        )

        return self.predictor.predict(
            feature_dataframe
        )

    def predict_latest(self) -> dict:
        """Predict the next-day return using the latest data."""

        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Feature dataset not found: {FEATURES_PATH}"
            )

        data = pd.read_csv(FEATURES_PATH)

        if data.empty:
            raise ValueError(
                "Feature dataset is empty."
            )

        latest = data.tail(1)

        prediction = self.predictor.predict(
            latest
        )

        return {
            "date": str(latest.iloc[0]["date"]),
            "predicted_return": prediction,
        }