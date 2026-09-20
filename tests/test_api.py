from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.services import prediction_service


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "xgboost"


def test_latest_prediction(tmp_path, monkeypatch):
    features = pd.DataFrame(
        [
            {
                "date": "2025-12-30",
                "return_1d": 0.01,
                "return_7d": 0.03,
                "sma_7": 85000,
                "sma_30": 83000,
                "ema_7": 85000,
                "ema_30": 83000,
                "volatility_7": 0.02,
                "volatility_30": 0.03,
                "volume_change": 0.10,
                "price_range": 0.05,
            }
        ]
    )

    features_path = (
        tmp_path / "btc-usd_features.csv"
    )

    features.to_csv(
        features_path,
        index=False,
    )

    monkeypatch.setattr(
        prediction_service,
        "FEATURES_PATH",
        Path(features_path),
    )

    response = client.get("/predict/latest")

    assert response.status_code == 200

    data = response.json()

    assert data["date"] == "2025-12-30"
    assert "predicted_return" in data

    assert isinstance(
        data["predicted_return"],
        float,
    )


def test_prediction_endpoint():
    payload = {
        "return_1d": 0.01,
        "return_7d": 0.03,
        "sma_7": 85000,
        "sma_30": 83000,
        "ema_7": 85000,
        "ema_30": 83000,
        "volatility_7": 0.02,
        "volatility_30": 0.03,
        "volume_change": 0.10,
        "price_range": 0.05,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_return" in data

    assert isinstance(
        data["predicted_return"],
        float,
    )