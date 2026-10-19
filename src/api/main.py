"""Step 11: FastAPI prediction service — one consistent /forecast endpoint.

Run:  uvicorn src.api.main:app --reload
Then open http://localhost:8000/docs for the interactive Swagger UI.

POST /forecast with a 168-hour history and an origin timestamp; returns 24 hourly
predictions in kW. Input is validated: wrong length or non-finite values are rejected.
"""
from datetime import datetime, timedelta
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import CFG
from src.predict import forecast, available_models, CONTEXT, HORIZON

app = FastAPI(title="Electricity Demand Forecast API", version="0.1.0")


class ForecastRequest(BaseModel):
    history: list[float] = Field(..., description=f"exactly {CONTEXT} hourly kW values")
    origin: datetime = Field(..., description="timestamp of the first predicted hour (ISO 8601)")
    model: str = Field("LightGBM", description="one of the available models")

    @field_validator("history")
    @classmethod
    def _check_history(cls, v):
        if len(v) != CONTEXT:
            raise ValueError(f"history must have exactly {CONTEXT} values, got {len(v)}")
        if not all(np.isfinite(v)):
            raise ValueError("history contains non-finite values")
        return v


class ForecastResponse(BaseModel):
    model: str
    client: str
    units: str
    timestamps: list[datetime]
    predictions: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "models": available_models()}


@app.post("/forecast", response_model=ForecastResponse)
def make_forecast(req: ForecastRequest):
    if req.model not in available_models():
        raise HTTPException(422, f"model '{req.model}' not available. "
                                 f"Train it first; have: {available_models()}")
    try:
        preds = forecast(req.history, req.origin, req.model)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except FileNotFoundError:
        raise HTTPException(503, "model artifacts missing — run `python -m src.train`")
    ts = [req.origin + timedelta(hours=j) for j in range(HORIZON)]
    return ForecastResponse(model=req.model, client=CFG["data"]["client"],
                            units="kW (hourly mean)", timestamps=ts,
                            predictions=[float(p) for p in preds])
