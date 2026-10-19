"""Step 8: Chronos zero-shot foundation model — NO training on our data.

We feed only the historic context at each origin and ask for 24 steps. We use the
Chronos-Bolt model (fast, CPU-friendly). It is labelled by its real model id; if the
exact 'Chronos-2' release is unavailable in the installed package we say so rather
than mislabel the result.
"""
import time
import numpy as np
import pandas as pd
import torch

from src.config import CFG, ROOT
from src.data.split import build as build_splits
from src.evaluate import aggregate
from src.results import log_result

MODEL_ID = "amazon/chronos-bolt-small"
LABEL = "Chronos-Bolt (small)"


def load_pipeline():
    try:
        from chronos import BaseChronosPipeline
        return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu",
                                                    torch_dtype=torch.float32)
    except Exception:
        from chronos import ChronosPipeline
        return ChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu",
                                                torch_dtype=torch.float32)


def point_forecast(pipe, context: torch.Tensor, hor: int) -> np.ndarray:
    # Bolt: predict_quantiles -> (quantiles[B,h,Q], mean[B,h])
    if hasattr(pipe, "predict_quantiles"):
        q, _ = pipe.predict_quantiles(context, prediction_length=hor,
                                      quantile_levels=[0.5])
        return q[0, :, 0].cpu().numpy()
    out = pipe.predict(context, prediction_length=hor)  # [B,S,h]
    return np.median(out[0].cpu().numpy(), axis=0)


def main():
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    values = s.to_numpy(np.float32)
    splits, _ = build_splits()
    ctx, hor = CFG["protocol"]["context"], CFG["protocol"]["horizon"]

    pipe = load_pipeline()
    print(f"[chronos] loaded {MODEL_ID}")

    test = splits["test"]
    trues, preds, lats = [], [], []
    for _, tgt, t in test:
        context = torch.tensor(values[t - ctx:t], dtype=torch.float32)
        t0 = time.perf_counter()
        pred = point_forecast(pipe, context, hor)
        lats.append((time.perf_counter() - t0) * 1000)
        preds.append(np.asarray(pred)[:hor])
        trues.append(tgt)

    m = aggregate(trues, preds)
    log_result({"model": LABEL, "split": "test", **m,
                "p50_latency_ms": round(float(np.median(lats)), 2),
                "notes": f"zero-shot; {MODEL_ID}"})
    np.savez(ROOT / "results" / "pred_chronos.npz",
             preds=np.stack(preds), origins=np.array([t for _, _, t in test]),
             label=LABEL)
    print(f"[chronos] done; p50 latency {np.median(lats):.0f} ms")


if __name__ == "__main__":
    main()
