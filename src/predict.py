"""Shared inference: turn a 168-hour history into a 24-hour forecast for any model.

Used by the FastAPI service (src/api/main.py). Mirrors the training-time feature
construction exactly so served forecasts match the benchmark.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import CFG, ROOT
from src.features.build import origin_features, calendar_features
from src.models.seasonal import seasonal_naive

MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
CONTEXT = CFG["protocol"]["context"]
HORIZON = CFG["protocol"]["horizon"]


def load_metadata() -> dict:
    with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def available_models() -> list[str]:
    models = ["Seasonal naive (24h)"]
    if (MODELS_DIR / "lgbm.joblib").exists():
        models.append("LightGBM")
    for cell in ("RNN", "LSTM", "GRU"):
        if (MODELS_DIR / f"{cell}.pt").exists():
            models.append(cell)
    return models


def forecast(history, origin_ts, model: str) -> np.ndarray:
    """history: 168 kW values ending just before origin_ts (the first predicted hour)."""
    history = np.asarray(history, dtype=np.float32)
    if len(history) != CONTEXT:
        raise ValueError(f"history must have exactly {CONTEXT} values, got {len(history)}")
    origin_ts = pd.Timestamp(origin_ts)

    if model.startswith("Seasonal"):
        return seasonal_naive(history, HORIZON, season=24)

    if model == "LightGBM":
        import joblib
        bundle = joblib.load(MODELS_DIR / "lgbm.joblib")
        models, cols = bundle["models"], bundle["columns"]
        base = origin_features(history, len(history))
        rows = [{**base, **calendar_features(origin_ts + pd.Timedelta(hours=j))}
                for j in range(HORIZON)]
        X_all = pd.DataFrame(rows)[cols]
        return np.array([models[j].predict(X_all.iloc[[j]])[0] for j in range(HORIZON)])

    if model in ("RNN", "LSTM", "GRU"):
        import torch
        from src.models.rnn import RecurrentForecaster
        ckpt = torch.load(MODELS_DIR / f"{model}.pt", map_location="cpu", weights_only=False)
        mu, sd = ckpt["mu"], ckpt["sd"]
        net = RecurrentForecaster(model)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        x = torch.tensor((history - mu) / sd, dtype=torch.float32).reshape(1, CONTEXT, 1)
        with torch.no_grad():
            return (net(x).numpy().ravel() * sd + mu)

    raise ValueError(f"unknown model: {model}")
