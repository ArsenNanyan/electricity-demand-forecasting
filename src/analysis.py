"""Step 10: error analysis — where do forecasts fail? Weekday/weekend + worst days.

Reads results/backtest.npz and the hourly series. Saves:
  artifacts/reports/failure_weekend.png   MAE weekday vs weekend per model
  results/failure_analysis.csv            segment MAEs + worst origins (best model)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CFG, ROOT

OUT = ROOT / CFG["paths"]["reports_dir"]


def main():
    bt = dict(np.load(ROOT / "results" / "backtest.npz", allow_pickle=True))
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    origins = bt["origins"]
    actuals = bt["actuals"]
    origin_ts = s.index[origins]
    is_weekend = origin_ts.dayofweek >= 5

    models = [k[len("pred_"):] for k in bt if k.startswith("pred_")]
    label = {m: m.replace("_", " ") for m in models}

    # per-window MAE for each model
    rows = []
    wmae = {}
    for m in models:
        err = np.abs(bt[f"pred_{m}"] - actuals).mean(axis=1)  # [n_origins]
        wmae[m] = err
        rows.append({"model": label[m],
                     "MAE_weekday": round(float(err[~is_weekend].mean()), 3),
                     "MAE_weekend": round(float(err[is_weekend].mean()), 3),
                     "MAE_overall": round(float(err.mean()), 3)})
    seg = pd.DataFrame(rows).sort_values("MAE_overall")

    # weekday vs weekend bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(seg)); w = 0.4
    ax.bar(x - w/2, seg["MAE_weekday"], w, label="weekday")
    ax.bar(x + w/2, seg["MAE_weekend"], w, label="weekend")
    ax.set_xticks(x); ax.set_xticklabels(seg["model"], rotation=30, ha="right")
    ax.set_ylabel("MAE (kW)"); ax.set_title("Where forecasts fail: weekday vs weekend")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "failure_weekend.png", dpi=100); plt.close(fig)

    # worst 5 origins for the best (lowest overall MAE) model
    best = seg.iloc[0]["model"]
    best_key = [m for m in models if label[m] == best][0]
    order = np.argsort(wmae[best_key])[::-1][:5]
    worst = pd.DataFrame({"origin": [str(origin_ts[i]) for i in order],
                          "weekend": [bool(is_weekend[i]) for i in order],
                          f"{best}_MAE_kw": [round(float(wmae[best_key][i]), 3) for i in order]})

    out_csv = ROOT / "results" / "failure_analysis.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("# Segment MAE by model\n")
        seg.to_csv(f, index=False)
        f.write(f"\n# Worst 5 forecast origins for best model ({best})\n")
        worst.to_csv(f, index=False)
    print(seg.to_string(index=False))
    print(f"\n[analysis] best model: {best}; worst days saved to {out_csv}")


if __name__ == "__main__":
    main()
