"""Step 2b: pick one client, aggregate to hourly MEAN kW, drop pre-activation zeros.

Writes data/processed/client_hourly.parquet (a single tidy series).
"""
import pandas as pd

from src.config import CFG, ROOT
from src.data.load import TXT_PATH


def build() -> pd.Series:
    c = CFG["data"]
    client = c["client"]
    # Semicolon-separated, comma decimal mark; first column is the timestamp.
    raw = pd.read_csv(TXT_PATH, sep=c["sep"], decimal=c["decimal"],
                      index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index)
    print(f"[clean] raw shape {raw.shape}, {raw.index.min()} -> {raw.index.max()}")

    # 15-min kW -> hourly MEAN kW (average power, NOT summed energy).
    s = raw[client].sort_index().resample(c["resample"]).mean().rename("load_kw")

    # Many clients have structural zeros before activation. Trim the leading
    # all-zero stretch so we model only the active period.
    first_active = s[s > 0].index.min()
    before = len(s)
    s = s.loc[first_active:]
    print(f"[clean] client {client}: dropped {before - len(s)} pre-activation hours; "
          f"active from {first_active}")
    print(f"[clean] hourly points: {len(s)}; NaNs: {int(s.isna().sum())}")

    out = ROOT / c["processed_file"]
    s.to_frame().to_parquet(out)
    print(f"[clean] wrote {out}")
    return s


if __name__ == "__main__":
    build()
