# Electricity Demand Forecasting — Sequence Modeling Benchmark

Predict the **next 24 hourly electricity-demand values** for one customer from historical
consumption, and compare forecasting methods *fairly* on identical data and forecast origins:

- **Seasonal naïve** (yesterday / last-week) — the baseline every model must beat
- **LightGBM / XGBoost** — gradient-boosted trees on lag + calendar features
- **RNN / LSTM / GRU** — recurrent neural nets (PyTorch)
- **PatchTST** — a time-series transformer *(optional stretch)*
- **Chronos-2 / TimesFM** — pretrained zero-shot foundation models *(optional stretch)*

## Dataset

**UCI ElectricityLoadDiagrams20112014** — 370 clients, 15-minute kW readings, 2011–2014.

- Landing page: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- DOI: 10.24432/C58C86
- License: **CC BY 4.0** — credit *Artur Trindade / UCI Machine Learning Repository*.

We select one client, aggregate four quarter-hour readings into an **hourly mean kW** value
(we average, never sum — summing would turn average power into energy, kWh). 2011–2014 values
are historic; this is a benchmarking exercise, not a statement about current demand.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m src.data.load          # download + verify dataset
python -m src.data.clean         # build hourly client parquet
python -m src.data.split         # chronological train/val/test + windows
pytest                           # leakage + shape safety tests
```

## The fair-comparison rule

Every method uses the **same client, same history length, same forecast origins, and the same
target representation**. Scalers / imputation / feature choices are fit on **training data only**;
the test split stays untouched until final evaluation. See `configs/default.yaml` for the exact
protocol constants.

## Layout

```
configs/            experiment protocol (single source of truth)
data/raw|processed|splits
notebooks/01_eda.ipynb
src/data|features|models        pipeline + model code
src/train.py  evaluate.py  predict.py
src/api/main.py  app/streamlit_app.py
tests/          leakage + shape tests
artifacts/  results/            saved models, reports, metrics
```

## Results

Client **MT_001**, hourly mean kW, test split = last 15% of the timeline (164 rolling
daily origins, each forecasting 24 hours). Context = 168 h. All numbers are measured on
identical test origins; none are invented.

| Model | Test MAE (kW) | RMSE (kW) | sMAPE (%) | p50 latency (ms) |
|---|---|---|---|---|
| **Chronos-Bolt (small)** — zero-shot | **1.218** | 2.771 | **31.4** | 33.2 |
| **PatchTST** | 1.306 | **2.682** | 35.0 | 31.2 |
| Seasonal naïve (24h) | 1.447 | 3.457 | 34.7 | ~0 |
| LSTM | 1.537 ±0.004 | 3.012 | 41.5 | 0.04 |
| GRU | 1.543 ±0.014 | 2.982 | 42.9 | 0.14 |
| RNN | 2.003 ±0.345 | 3.376 | 51.5 | 0.04 |
| LightGBM | 2.010 | 3.272 | 51.7 | 24.7 |
| Seasonal naïve (168h) | 2.261 | 4.900 | 45.0 | ~0 |

`±` is the standard deviation of MAE across 3 training seeds. See
`artifacts/reports/compare_bars.png` and `compare_horizon.png`.

### When complex models beat (and fail to beat) the baseline

- **The two modern architectures are the only models that beat the baseline.** A
  **zero-shot foundation model (Chronos-Bolt)** — trained on *other* series, never on this
  client — is the best on MAE and sMAPE, and the **PatchTST transformer** has the lowest
  RMSE. This is the headline: pretrained / attention-based models clearly outperform here.
- **The naïve baseline still beats all the *classic* trained models.** "Same hour
  yesterday" (1.45 kW) beats LSTM, GRU, RNN and LightGBM on MAE. For a single, highly
  periodic commercial client, a well-chosen baseline is genuinely strong — a trained model
  is not automatically better, and saying so honestly is the point of the benchmark.
- **Among the recurrent nets, gating matters.** LSTM/GRU beat the vanilla RNN and win on
  RMSE vs the baseline (≈3.0 vs 3.46 — fewer big misses). The RNN is *unstable* across
  seeds (±0.345 vs ±0.004), the textbook vanilla-RNN gradient limitation. **GRU ≈ LSTM**
  with fewer parameters.
- **LightGBM** trailed on MAE/sMAPE but still beat the naïve on RMSE; richer features or
  Optuna tuning would likely help.
- **Cost vs quality:** Chronos/PatchTST cost ~30 ms per forecast and (for Chronos) a model
  download; the RNNs run in microseconds. The best accuracy here is not the cheapest.
- **Error vs horizon** (`compare_horizon.png`) is not monotonic — it peaks in the evening
  hours (~h17, h19), reflecting this client's daily demand shape rather than pure drift.

### Reproduce the results

```bash
python -m src.data.load && python -m src.data.clean && python -m src.data.split
python -m src.models.run_seasonal   # baselines
python -m src.models.gbm            # LightGBM
python -m src.models.rnn            # RNN / LSTM / GRU (3 seeds)
python -m src.backtest              # precompute dashboard forecasts
python -m src.plots                 # comparison charts
streamlit run app/streamlit_app.py  # interactive demo at http://localhost:8501
pytest                              # leakage + shape safety tests
```

### Limitations

Single client, 2011–2014 historic data, CPU-only training, backtest evaluation only.
Not a statement about current demand. The foundation-model row uses **Chronos-Bolt
(small)** as the available pretrained model, labelled by its real id rather than assuming
a specific "Chronos-2" release. **TimesFM** is not included — note its pretrained weights
are non-commercial, so check licensing before any hosted/commercial demo.
