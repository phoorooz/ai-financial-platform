from fastapi import FastAPI
from pydantic import BaseModel

from app.services.prediction_service import PredictionService


app = FastAPI(
    title="AI Financial Forecasting API",
    version="1.0.0",
    description="Bitcoin next-day return prediction API.",
)


prediction_service = PredictionService()


class PredictionRequest(BaseModel):
    return_1d: float
    return_7d: float
    sma_7: float
    sma_30: float
    ema_7: float
    ema_30: float
    volatility_7: float
    volatility_30: float
    volume_change: float
    price_range: float


class PredictionResponse(BaseModel):
    predicted_return: float


class LatestPredictionResponse(BaseModel):
    date: str
    predicted_return: float


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "xgboost",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    prediction = prediction_service.predict(
        request.model_dump()
    )

    return PredictionResponse(
        predicted_return=prediction
    )


@app.get(
    "/predict/latest",
    response_model=LatestPredictionResponse,
)
def predict_latest():
    result = prediction_service.predict_latest()

    return LatestPredictionResponse(
        **result
    )