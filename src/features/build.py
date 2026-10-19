"""Step 5 features for tree models — all computed from data available AT the origin.

For an origin t (forecast start) we build, for each horizon j (0..23):
  - lag features: values at t-1, t-2, t-3, t-24, t-48, t-168  (last known at t)
  - rolling mean/std over the 24h and 168h BEFORE t                (no leakage)
  - calendar of the TARGET timestamp t+j: hour, weekday, weekend, month (known ahead)
Target = values[t+j].
"""
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 24, 48, 168]
ROLLS = [24, 168]


def origin_features(values: np.ndarray, t: int) -> dict:
    """Features known at origin t (shared across all horizons)."""
    f = {f"lag_{L}": float(values[t - L]) for L in LAGS}
    for w in ROLLS:
        window = values[t - w:t]
        f[f"rmean_{w}"] = float(window.mean())
        f[f"rstd_{w}"] = float(window.std())
    return f


def calendar_features(ts: pd.Timestamp) -> dict:
    return {
        "hour": ts.hour,
        "weekday": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "month": ts.month,
    }


def make_table(values, index, origins, horizon=24):
    """Return dict: horizon j -> (X DataFrame, y array) over the given origins."""
    per_h = {j: ([], []) for j in range(horizon)}
    for t in origins:
        base = origin_features(values, t)
        for j in range(horizon):
            row = {**base, **calendar_features(index[t + j])}
            per_h[j][0].append(row)
            per_h[j][1].append(float(values[t + j]))
    return {j: (pd.DataFrame(X), np.asarray(y, float)) for j, (X, y) in per_h.items()}
