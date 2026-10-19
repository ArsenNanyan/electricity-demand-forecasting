import numpy as np
from src.models.seasonal import seasonal_naive
from src.evaluate import forecast_metrics


def test_seasonal_naive_shape_and_repeat():
    hist = np.arange(336, dtype=float)  # two weeks
    pred = seasonal_naive(hist, horizon=24, season=168)
    assert pred.shape == (24,)
    # last-week naive repeats the value exactly 168h before each target step
    assert pred[0] == hist[-168]


def test_metrics_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    m = forecast_metrics(y, y)
    assert m["MAE_kw"] == 0 and m["RMSE_kw"] == 0 and m["sMAPE_pct"] < 1e-4
