"""Step 9: one metrics function used by EVERY model, so comparisons are fair."""
import numpy as np


def forecast_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ae = np.abs(y - pred)
    return {
        "MAE_kw": float(ae.mean()),
        "RMSE_kw": float(np.sqrt(((y - pred) ** 2).mean())),
        # symmetric MAPE in %; +1e-8 guards the near-zero-demand case.
        "sMAPE_pct": float(200 * (ae / (np.abs(y) + np.abs(pred) + 1e-8)).mean()),
    }


def aggregate(windows_true, windows_pred) -> dict:
    """Flatten a list of 24-step forecasts into one global metric set."""
    y = np.concatenate([np.asarray(w).ravel() for w in windows_true])
    p = np.concatenate([np.asarray(w).ravel() for w in windows_pred])
    return forecast_metrics(y, p)
