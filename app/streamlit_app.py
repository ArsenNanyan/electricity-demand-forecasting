"""Step 12: Electricity Forecast Lab — compare 24h demand forecasts on real backtests.

Run:  streamlit run app/streamlit_app.py
Loads precomputed backtest forecasts (results/backtest.npz) produced by src.backtest.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "results" / "backtest.npz"
SERIES = ROOT / "data" / "processed" / "client_hourly.parquet"
METRICS = ROOT / "results" / "metrics.csv"
HORIZON_ERR = ROOT / "results" / "horizon_error.csv"
CLIENT = "MT_001"

st.set_page_config(page_title="Electricity Forecast Lab", layout="wide")
st.title("24-Hour Electricity Demand Forecast")
st.caption(
    f"Client **{CLIENT}** · units **kW (hourly mean)** · source: UCI "
    "ElectricityLoadDiagrams20112014 (CC BY 4.0, Artur Trindade / UCI). "
    "Values are 2011–2014 historic data — a benchmark, not current demand."
)

if not NPZ.exists():
    st.error("No backtest found. Run:  `python -m src.backtest`  first.")
    st.stop()


@st.cache_data
def load():
    d = dict(np.load(NPZ, allow_pickle=True))
    s = pd.read_parquet(SERIES)["load_kw"]
    metrics = pd.read_csv(METRICS).drop_duplicates("model", keep="last")
    hmae = pd.read_csv(HORIZON_ERR, index_col=0) if HORIZON_ERR.exists() else None
    return d, s, metrics, hmae


d, series, metrics, hmae = load()
models = [k[len("pred_"):] for k in d if k.startswith("pred_")]
label = {k: k.replace("_", " ") for k in models}
origins = d["origins"]
origin_times = series.index[origins]

# ---- Controls ----
c1, c2 = st.columns([2, 3])
with c1:
    picked = st.multiselect("Models to compare", models,
                            default=[m for m in ("LSTM", "GRU", "Seasonal_naive_24h")
                                     if m in models] or models[:2],
                            format_func=lambda k: label[k])
with c2:
    i = st.select_slider("Backtest forecast origin",
                         options=list(range(len(origins))),
                         value=len(origins) // 2,
                         format_func=lambda j: str(origin_times[j]))

if not picked:
    st.info("Select at least one model above.")
    st.stop()

origin_t = int(origins[i])
origin_ts = origin_times[i]
future_idx = series.index[origin_t:origin_t + 24]
actual = d["actuals"][i]

# ---- (3) historical context: last 7 days + (2) 24h forecast ----
hist = series.iloc[max(0, origin_t - 168):origin_t]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="history (7d)",
                         line=dict(color="#888", width=1)))
fig.add_trace(go.Scatter(x=future_idx, y=actual, name="actual (next 24h)",
                         line=dict(color="black", width=3)))
for m in picked:
    fig.add_trace(go.Scatter(x=future_idx, y=d[f"pred_{m}"][i],
                             name=label[m], mode="lines+markers"))
fig.add_vline(x=origin_ts, line_dash="dash", line_color="red")
fig.update_layout(title=f"Forecast origin {origin_ts}  (red = now)",
                  yaxis_title="kW", height=420, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ---- per-origin error + (7) CSV download ----
rows = {"timestamp": future_idx, "actual_kw": actual}
for m in picked:
    rows[f"{label[m]}_kw"] = d[f"pred_{m}"][i]
table = pd.DataFrame(rows)
e1, e2 = st.columns([3, 1])
with e1:
    st.subheader("This origin — mean abs error (kW)")
    st.write({label[m]: round(float(np.abs(d[f"pred_{m}"][i] - actual).mean()), 3)
              for m in picked})
with e2:
    st.download_button("Download forecast CSV",
                       table.to_csv(index=False).encode(),
                       f"forecast_{origin_ts:%Y%m%d_%H}.csv", "text/csv")

# ---- (4) overall metrics table ----
st.subheader("Overall test metrics (identical test origins)")
show = metrics[["model", "MAE_kw", "RMSE_kw", "sMAPE_pct", "p50_latency_ms"]]
st.dataframe(show.round(3), use_container_width=True, hide_index=True)
st.caption("Lower is better. Seasonal naïve is the baseline every model must beat.")

# ---- (5) horizon error + (6) latency ----
g1, g2 = st.columns(2)
with g1:
    if hmae is not None:
        hf = go.Figure()
        for m in picked:
            col = label[m]
            if col in hmae.columns:
                hf.add_trace(go.Scatter(x=hmae.index + 1, y=hmae[col], name=col))
        hf.update_layout(title="Error by horizon step", xaxis_title="hours ahead",
                         yaxis_title="MAE (kW)", height=330)
        st.plotly_chart(hf, use_container_width=True)
with g2:
    lat = metrics.dropna(subset=["p50_latency_ms"])
    lf = go.Figure(go.Bar(x=lat["model"], y=lat["p50_latency_ms"]))
    lf.update_layout(title="p50 latency per 24-step forecast",
                     yaxis_title="ms", height=330)
    st.plotly_chart(lf, use_container_width=True)

# ---- (8) messages for optional/absent models ----
for opt in ("PatchTST", "Chronos-2"):
    if not any(opt.replace("-", "") in m.replace("-", "") for m in models):
        st.info(f"{opt} not trained yet — optional stretch model (Steps 7–8).")

st.caption("Limitation: single client, 2011–2014 historic data, backtest only. "
           "Forecasts beyond dataset history would show predictions without ground truth.")
