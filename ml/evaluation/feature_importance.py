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


def load_model() -> XGBRegressor:
    """Load trained XGBoost model."""

    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    return model


def main() -> None:

    print("Loading XGBoost model...")

    model = load_model()

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print("\nFeature Importance")
    print("------------------")

    for _, row in importance.iterrows():
        print(
            f"{row['feature']:<20}"
            f"{row['importance']:.6f}"
        )


if __name__ == "__main__":
    main()