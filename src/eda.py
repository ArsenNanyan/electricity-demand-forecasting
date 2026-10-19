"""Step 2: EDA — save diagnostic plots and print key stats for the chosen client."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CFG, ROOT

OUT = ROOT / CFG["paths"]["reports_dir"]


def main():
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    print(f"[eda] range {s.index.min()} -> {s.index.max()}, n={len(s)}")
    print(f"[eda] kW  mean={s.mean():.1f}  min={s.min():.1f}  max={s.max():.1f}  "
          f"zeros={(s == 0).sum()}")

    # 1. Full timeline
    fig, ax = plt.subplots(figsize=(12, 3))
    s.plot(ax=ax, lw=0.4)
    ax.set(title=f"{CFG['data']['client']} — full hourly demand", ylabel="kW")
    fig.tight_layout(); fig.savefig(OUT / "eda_timeline.png", dpi=90); plt.close(fig)

    # 2. One representative week (first full week of summer)
    week = s.loc["2013-07-01":"2013-07-07"]
    fig, ax = plt.subplots(figsize=(12, 3))
    week.plot(ax=ax)
    ax.set(title="Representative week (2013-07-01..07)", ylabel="kW")
    fig.tight_layout(); fig.savefig(OUT / "eda_week.png", dpi=90); plt.close(fig)

    # 3. Hour x day-of-week heatmap (average daily/weekly shape)
    df = s.to_frame()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    piv = df.pivot_table("load_kw", "hour", "dow", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(piv, cmap="viridis", ax=ax,
                cbar_kws={"label": "mean kW"})
    ax.set(title="Mean demand by hour x weekday", xlabel="weekday (0=Mon)", ylabel="hour")
    fig.tight_layout(); fig.savefig(OUT / "eda_heatmap.png", dpi=90); plt.close(fig)

    # 4. Autocorrelation up to 2 weeks
    x = s.to_numpy() - s.mean()
    acf = np.correlate(x, x, "full")[len(x) - 1:]
    acf = acf[:24 * 14] / acf[0]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(acf); ax.axvline(24, color="r", ls="--", lw=.8, label="24h")
    ax.axvline(168, color="g", ls="--", lw=.8, label="168h")
    ax.set(title="Autocorrelation", xlabel="lag (hours)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "eda_acf.png", dpi=90); plt.close(fig)

    print(f"[eda] saved 4 plots to {OUT}")


if __name__ == "__main__":
    main()
