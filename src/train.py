"""Train the servable models and save artifacts (+ optional MLflow tracking).

Saves to artifacts/models/:
  lgbm.joblib          24 LightGBM regressors + feature column order
  RNN.pt / LSTM.pt / GRU.pt   state_dict + train scaler (mu, sd)
  metadata.json        client, context, horizon, scaler, model list
"""
import json
import numpy as np
import pandas as pd
import joblib
import torch
import lightgbm as lgb

from src.config import CFG, ROOT
from src.data.split import build as build_splits
from src.features.build import make_table
from src.models.gbm import PARAMS
from src.models.rnn import to_tensors, train_one

MODELS_DIR = ROOT / CFG["paths"]["models_dir"]


def main():
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    values = s.to_numpy(np.float32)
    index = s.index
    splits, meta = build_splits()
    hor = CFG["protocol"]["horizon"]

    try:
        import mlflow
        mlflow.set_experiment("electricity-forecast-benchmark")
        run = mlflow.start_run()
        mlflow.log_params({"client": CFG["data"]["client"],
                           "context": CFG["protocol"]["context"], "horizon": hor,
                           "seed": CFG["seed"]})
    except Exception:
        mlflow = None

    # LightGBM — 24 direct models
    tr = make_table(values, index, [t for _, _, t in splits["train"]], hor)
    gbm = {j: lgb.LGBMRegressor(**PARAMS).fit(*tr[j]) for j in range(hor)}
    columns = list(tr[0][0].columns)
    joblib.dump({"models": gbm, "columns": columns}, MODELS_DIR / "lgbm.joblib")
    print("[train] saved lgbm.joblib")

    # RNN / LSTM / GRU (seed 42)
    mu = float(values[:meta["train_end"]].mean())
    sd = float(values[:meta["train_end"]].std())
    Xtr, Ytr = to_tensors(splits["train"], mu, sd)
    Xva, Yva = to_tensors(splits["valid"], mu, sd)
    for cell in ("RNN", "LSTM", "GRU"):
        net = train_one(cell, 42, (Xtr, Ytr, Xva, Yva), mu, sd)
        torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd},
                   MODELS_DIR / f"{cell}.pt")
        print(f"[train] saved {cell}.pt")

    metadata = {"client": CFG["data"]["client"], "context": CFG["protocol"]["context"],
                "horizon": hor, "mu": mu, "sd": sd, "columns": columns,
                "models": ["Seasonal naive (24h)", "LightGBM", "RNN", "LSTM", "GRU"],
                "units": "kW (hourly mean)"}
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("[train] saved metadata.json")

    if mlflow is not None:
        try:
            m = pd.read_csv(ROOT / "results" / "metrics.csv").drop_duplicates("model", keep="last")
            for _, r in m.iterrows():
                tag = r["model"].replace(" ", "_").replace("(", "").replace(")", "")
                mlflow.log_metric(f"MAE_{tag}", float(r["MAE_kw"]))
            mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")
            mlflow.end_run()
            print("[train] logged run to MLflow (mlruns/)")
        except Exception:
            pass


if __name__ == "__main__":
    main()
