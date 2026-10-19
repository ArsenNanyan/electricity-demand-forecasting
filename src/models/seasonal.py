"""Step 4: seasonal-naive baselines — the threshold every model must beat.

season=24  -> "same hour yesterday"
season=168 -> "same hour last week"
"""
import numpy as np


def seasonal_naive(history: np.ndarray, horizon: int = 24, season: int = 168) -> np.ndarray:
    assert len(history) >= season, "need at least `season` hours of history"
    return np.array([history[-season + (j % season)] for j in range(horizon)],
                    dtype=float)
