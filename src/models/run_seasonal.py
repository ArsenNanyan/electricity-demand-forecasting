"""Step 4 runner: evaluate seasonal-naive baselines; pick season on validation."""
import numpy as np
import pandas as pd

from src.config import CFG, ROOT
from src.data.split import build as build_splits
from src.models.seasonal import seasonal_naive
from src.evaluate import aggregate
from src.results import log_result


def eval_split(values, windows, season):
    trues, preds = [], []
    for _, tgt, t in windows:
        hist = values[:t]                      # everything before the origin
        preds.append(seasonal_naive(hist, horizon=len(tgt), season=season))
        trues.append(tgt)
    return aggregate(trues, preds)


def main():
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    values = s.to_numpy(np.float32)
    splits, _ = build_splits()

    # Choose season (24 vs 168) on VALIDATION only.
    print("\n[seasonal] validation (choose season):")
    val = {sea: eval_split(values, splits["valid"], sea) for sea in (24, 168)}
    for sea, m in val.items():
        print(f"  season={sea}: MAE={m['MAE_kw']:.3f}  sMAPE={m['sMAPE_pct']:.1f}")
    best = min(val, key=lambda sea: val[sea]["MAE_kw"])
    print(f"[seasonal] best on val: season={best}")

    # Report BOTH on test (table needs both rows), labelled.
    for sea in (24, 168):
        m = eval_split(values, splits["test"], sea)
        log_result({"model": f"Seasonal naive ({sea}h)", "split": "test", **m,
                    "notes": "chosen" if sea == best else ""})


if __name__ == "__main__":
    main()
