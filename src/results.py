"""Shared results logger — every model appends one row so tables stay comparable."""
import csv
from pathlib import Path

from src.config import CFG, ROOT

RESULTS_CSV = ROOT / CFG["paths"]["results_dir"] / "metrics.csv"
FIELDS = ["model", "split", "MAE_kw", "RMSE_kw", "sMAPE_pct", "p50_latency_ms", "notes"]


def log_result(row: dict):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_CSV.exists()
    full = {k: row.get(k, "") for k in FIELDS}
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(full)
    print(f"[results] logged {full['model']} ({full['split']}): "
          f"MAE={full['MAE_kw']}  RMSE={full['RMSE_kw']}  sMAPE={full['sMAPE_pct']}")
