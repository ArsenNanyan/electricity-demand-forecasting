"""Step 3: chronological 70/15/15 split + rolling-origin windows.

CRUCIAL: the split is by TIME, never shuffled. A window's targets must live
entirely in its own split; its history may reach back into earlier data
(that is realistic — at inference you always know the recent past).
Saves index metadata to data/splits/ for inspection and tests.
"""
import json
import numpy as np
import pandas as pd

from src.config import CFG, ROOT
from src.data.windows import make_windows


def build():
    c, p = CFG["data"], CFG["protocol"]
    s = pd.read_parquet(ROOT / c["processed_file"])["load_kw"]
    values = s.to_numpy(dtype=np.float32)
    n = len(values)

    train_end = int(n * p["train_frac"])
    val_end = int(n * (p["train_frac"] + p["val_frac"]))
    ctx, hor, step = p["context"], p["horizon"], p["origin_step"]

    splits = {
        "train": make_windows(values, ctx, train_end, ctx, hor, step),
        "valid": make_windows(values, train_end, val_end, ctx, hor, step),
        "test":  make_windows(values, val_end, n, ctx, hor, step),
    }
    meta = {k: {"n_windows": len(v), "origins": [int(t) for _, _, t in v]}
            for k, v in splits.items()}
    meta.update({"n": n, "train_end": train_end, "val_end": val_end,
                 "context": ctx, "horizon": hor, "step": step})

    out = ROOT / CFG["paths"]["splits_dir"] / "split_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[split] n={n} train_end={train_end} val_end={val_end}")
    for k in ("train", "valid", "test"):
        print(f"[split] {k}: {meta[k]['n_windows']} windows")
    print(f"[split] wrote {out}")
    return splits, meta


if __name__ == "__main__":
    build()
