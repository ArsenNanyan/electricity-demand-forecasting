import numpy as np
from src.data.windows import make_windows


def test_shapes_and_alignment():
    values = np.arange(1000, dtype=np.float32)
    ctx, hor = 168, 24
    wins = make_windows(values, ctx, 500, ctx, hor, step=24)
    assert wins, "should produce windows"
    for hist, tgt, t in wins:
        assert len(hist) == ctx and len(tgt) == hor
        # history is strictly the ctx values before origin t
        assert np.array_equal(hist, values[t - ctx:t])
        assert np.array_equal(tgt, values[t:t + hor])


def test_targets_stay_in_range():
    values = np.arange(1000, dtype=np.float32)
    wins = make_windows(values, 300, 600, 168, 24, step=24)
    for _, _, t in wins:
        assert t + 24 <= 600  # target never spills past the split end
