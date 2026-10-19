"""Final comparison plots for the README: model bar chart + per-horizon error."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CFG, ROOT

OUT = ROOT / CFG["paths"]["reports_dir"]


def main():
    m = pd.read_csv(ROOT / "results" / "metrics.csv").drop_duplicates("model", keep="last")
    m = m.sort_values("MAE_kw")

    # 1. MAE vs RMSE grouped bars
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(m)); w = 0.4
    ax.bar(x - w/2, m["MAE_kw"], w, label="MAE (kW)")
    ax.bar(x + w/2, m["RMSE_kw"], w, label="RMSE (kW)")
    base = m.loc[m["model"] == "Seasonal naive (24h)", "MAE_kw"]
    if len(base):
        ax.axhline(float(base.iloc[0]), color="red", ls="--", lw=1,
                   label="baseline MAE")
    ax.set_xticks(x); ax.set_xticklabels(m["model"], rotation=30, ha="right")
    ax.set_ylabel("kW"); ax.set_title("Model comparison — MAE & RMSE (lower is better)")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "compare_bars.png", dpi=100); plt.close(fig)

    # 2. Error by horizon
    he = pd.read_csv(ROOT / "results" / "horizon_error.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    for col in he.columns:
        ax.plot(he.index + 1, he[col], marker=".", label=col)
    ax.set_xlabel("hours ahead"); ax.set_ylabel("MAE (kW)")
    ax.set_title("Error grows with horizon"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "compare_horizon.png", dpi=100); plt.close(fig)
    print(f"[plots] saved compare_bars.png, compare_horizon.png to {OUT}")


if __name__ == "__main__":
    main()
