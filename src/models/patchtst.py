"""Step 7: PatchTST transformer via NeuralForecast — direct 24h forecast.

PatchTST splits the 168h context into patches and attends over them. We train it
as one more supervised model on the SAME train range and evaluate on the SAME test
origins, so the comparison stays fair.
"""
import time
import warnings
import numpy as np
import pandas as pd

from src.config import CFG, ROOT
from src.data.split import build as build_splits
from src.evaluate import aggregate
from src.results import log_result

warnings.filterwarnings("ignore")


def main():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST

    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    values = s.to_numpy(np.float32)
    index = s.index
    splits, meta = build_splits()
    ctx, hor = CFG["protocol"]["context"], CFG["protocol"]["horizon"]

    # NeuralForecast wants a long dataframe: unique_id, ds, y. Train on the
    # training range only (up to train_end).
    train_end = meta["train_end"]
    df = pd.DataFrame({"unique_id": "MT_001",
                       "ds": index[:train_end],
                       "y": values[:train_end]})

    model = PatchTST(h=hor, input_size=ctx, max_steps=200,
                     scaler_type="standard", random_seed=CFG["seed"],
                     batch_size=32)
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(df)
    print("[patchtst] trained")

    # Predict at each test origin: feed history ending exactly at the origin.
    test = splits["test"]
    trues, preds, lats = [], [], []
    for _, tgt, t in test:
        hist_df = pd.DataFrame({"unique_id": "MT_001",
                                "ds": index[t - ctx:t],
                                "y": values[t - ctx:t]})
        t0 = time.perf_counter()
        out = nf.predict(df=hist_df)
        lats.append((time.perf_counter() - t0) * 1000)
        preds.append(out["PatchTST"].to_numpy()[:hor])
        trues.append(tgt)

    m = aggregate(trues, preds)
    log_result({"model": "PatchTST", "split": "test", **m,
                "p50_latency_ms": round(float(np.median(lats)), 2),
                "notes": f"NeuralForecast; context={ctx}"})
    # Save predictions for the dashboard.
    np.savez(ROOT / "results" / "pred_patchtst.npz",
             preds=np.stack(preds), origins=np.array([t for _, _, t in test]))
    print(f"[patchtst] done; p50 latency {np.median(lats):.0f} ms")


if __name__ == "__main__":
    main()
