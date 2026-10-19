"""Shared window generator — the heart of leakage-free evaluation.

An "origin" t is the START index of the forecast. The model may use the
`context` hours strictly BEFORE t (values[t-context:t]) and must predict
values[t:t+horizon]. Origins are spaced `step` apart (one day = 24h).
"""
import numpy as np


def make_windows(values: np.ndarray, start: int, stop: int,
                 context: int, horizon: int, step: int = 24):
    """Yield (history, target, origin) tuples with targets fully inside [start, stop)."""
    out = []
    for t in range(max(start, context), stop - horizon + 1, step):
        out.append((values[t - context:t], values[t:t + horizon], t))
    return out
