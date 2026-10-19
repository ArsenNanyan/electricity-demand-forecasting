import numpy as np
from src.data.windows import make_windows


def test_no_target_leakage_across_splits():
    """A split's forecast targets must lie entirely inside that split's range."""
    values = np.arange(2000, dtype=np.float32)
    n = len(values)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    ctx, hor, step = 168, 24, 24

    valid = make_windows(values, train_end, val_end, ctx, hor, step)
    test = make_windows(values, val_end, n, ctx, hor, step)

    # Validation targets never reach into the test region.
    for _, _, t in valid:
        assert t + hor <= val_end
    # Test targets never reach before the test region starts.
    for _, _, t in test:
        assert t >= val_end


def test_history_may_use_past_but_not_future():
    values = np.arange(2000, dtype=np.float32)
    wins = make_windows(values, 400, 800, 168, 24, step=24)
    for hist, tgt, t in wins:
        assert hist.max() < values[t]       # history is all before origin
        assert tgt[0] == values[t]          # prediction starts exactly at origin
