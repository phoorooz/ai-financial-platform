from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model"] == "xgboost"


def test_latest_prediction():
    response = client.get("/predict/latest")

    assert response.status_code == 200

    data = response.json()

    assert "date" in data
    assert "predicted_return" in data

    assert isinstance(
        data["date"],
        str,
    )

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