"""Demo GIF: the 24h forecast rolling across consecutive backtest days."""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CFG, ROOT

OUT = ROOT / CFG["paths"]["reports_dir"] / "demo.gif"
SHOW = {"Chronos-Bolt_small": "#2ca02c", "LSTM": "#d62728",
        "Seasonal_naive_24h": "#1f77b4"}


def main():
    bt = dict(np.load(ROOT / "results" / "backtest.npz", allow_pickle=True))
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    origins = bt["origins"]; actuals = bt["actuals"]
    start = len(origins) // 2
    frames = []
    for i in range(start, start + 14):
        t = int(origins[i])
        hist = s.iloc[t - 48:t]
        fut = s.index[t:t + 24]
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=80)
        ax.plot(hist.index, hist.values, color="#888", lw=1, label="history")
        ax.plot(fut, actuals[i], color="black", lw=2.5, label="actual")
        for key, col in SHOW.items():
            if f"pred_{key}" in bt:
                ax.plot(fut, bt[f"pred_{key}"][i], color=col, lw=1.5,
                        label=key.replace("_", " "))
        ax.axvline(s.index[t], color="red", ls="--", lw=0.8)
        ax.set_title(f"24h forecast — origin {s.index[t]:%Y-%m-%d %H:%M}")
        ax.set_ylabel("kW"); ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(0, float(max(actuals[i].max(), hist.values.max())) * 1.15)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        buf.seek(0); frames.append(Image.open(buf).convert("P"))
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=700, loop=0)
    print(f"[gif] saved {len(frames)} frames to {OUT}")


if __name__ == "__main__":
    main()
