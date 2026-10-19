"""Step 5 runner: 24 direct LightGBM regressors (one per horizon)."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import CFG, ROOT
from src.data.split import build as build_splits
from src.features.build import make_table
from src.evaluate import aggregate
from src.results import log_result

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
              random_state=CFG["seed"], n_jobs=-1, verbose=-1)


def main():
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    values = s.to_numpy(np.float32)
    index = s.index
    splits, _ = build_splits()
    hor = CFG["protocol"]["horizon"]

    train_origins = [t for _, _, t in splits["train"]]
    test_origins = [t for _, _, t in splits["test"]]
    tr = make_table(values, index, train_origins, hor)
    te = make_table(values, index, test_origins, hor)

    # Train one model per horizon.
    models = {}
    for j in range(hor):
        Xtr, ytr = tr[j]
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(Xtr, ytr)
        models[j] = m
    print(f"[gbm] trained {hor} horizon models")

    # Predict test origin-by-origin so we can measure per-forecast latency.
    trues, preds, latencies = [], [], []
    n_test = len(test_origins)
    for i in range(n_test):
        start = time.perf_counter()
        pred = np.array([models[j].predict(te[j][0].iloc[[i]])[0] for j in range(hor)])
        latencies.append((time.perf_counter() - start) * 1000)
        preds.append(pred)
        trues.append(np.array([te[j][1][i] for j in range(hor)]))

    m = aggregate(trues, preds)
    p50 = float(np.median(latencies))
    log_result({"model": "LightGBM", "split": "test", **m,
                "p50_latency_ms": round(p50, 2),
                "notes": "24 direct regressors; lags+rolling+calendar"})
    print(f"[gbm] p50 latency per 24-step forecast: {p50:.1f} ms")


if __name__ == "__main__":
    main()
