"""Precompute every model's 24h forecast at each TEST origin, save for the dashboard.

Output: results/backtest.npz with
  origins    : int origin indices into the hourly series
  actuals    : [n, 24] true kW
  pred_<model>: [n, 24] forecast kW
  hmae_<model>: [24] mean abs error by horizon step
and results/horizon_error.csv (tidy, per-horizon MAE per model).
"""
import numpy as np
import pandas as pd
import torch

from src.config import CFG, ROOT
from src.data.split import build as build_splits
from src.models.seasonal import seasonal_naive
from src.features.build import make_table
from src.models.rnn import to_tensors, train_one, RecurrentForecaster  # noqa: F401
import lightgbm as lgb
from src.models.gbm import PARAMS


def main():
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    values = s.to_numpy(np.float32)
    index = s.index
    splits, meta = build_splits()
    hor = CFG["protocol"]["horizon"]
    test = splits["test"]
    origins = [t for _, _, t in test]
    actuals = np.stack([y for _, y, _ in test])
    preds = {}

    # Seasonal naive (24h)
    preds["Seasonal naive (24h)"] = np.stack(
        [seasonal_naive(values[:t], hor, season=24) for t in origins])

    # LightGBM — 24 direct models
    tr = make_table(values, index, [t for _, _, t in splits["train"]], hor)
    te = make_table(values, index, origins, hor)
    gbm = {j: lgb.LGBMRegressor(**PARAMS).fit(*tr[j]) for j in range(hor)}
    preds["LightGBM"] = np.column_stack([gbm[j].predict(te[j][0]) for j in range(hor)])

    # RNN / LSTM / GRU — one seed (42) each for the demo
    mu = float(values[:meta["train_end"]].mean())
    sd = float(values[:meta["train_end"]].std())
    Xtr, Ytr = to_tensors(splits["train"], mu, sd)
    Xva, Yva = to_tensors(splits["valid"], mu, sd)
    Xte, _ = to_tensors(test, mu, sd)
    for cell in ("RNN", "LSTM", "GRU"):
        model = train_one(cell, 42, (Xtr, Ytr, Xva, Yva), mu, sd)
        model.eval()
        with torch.no_grad():
            preds[cell] = model(Xte).cpu().numpy() * sd + mu

    # Per-horizon MAE for each model
    hmae = {m: np.abs(p - actuals).mean(axis=0) for m, p in preds.items()}

    out = {"origins": np.array(origins), "actuals": actuals}
    for m, p in preds.items():
        key = m.replace(" ", "_").replace("(", "").replace(")", "")
        out[f"pred_{key}"] = p
        out[f"hmae_{key}"] = hmae[m]
    np.savez(ROOT / "results" / "backtest.npz", **out)

    tidy = pd.DataFrame(hmae); tidy.index.name = "horizon_step"
    tidy.to_csv(ROOT / "results" / "horizon_error.csv")
    print(f"[backtest] saved {len(origins)} origins x {hor}h for {len(preds)} models")


if __name__ == "__main__":
    main()
