"""Step 6: RNN / LSTM / GRU direct 24-output forecasters (PyTorch, CPU-friendly).

Crucial fairness details:
  - standardize with TRAIN statistics only; invert before reporting kW errors
  - early stopping on validation; identical config across the three cells
  - repeat over seeds and report mean +/- std (neural nets are stochastic)
"""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import CFG, ROOT
from src.data.split import build as build_splits
from src.evaluate import aggregate
from src.results import log_result

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN, MAX_EPOCHS, PATIENCE, BATCH = 64, 100, 10, 64
SEEDS = [42, 43, 44]


class RecurrentForecaster(nn.Module):
    def __init__(self, cell="GRU", hidden=HIDDEN, horizon=24):
        super().__init__()
        klass = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
        self.encoder = klass(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):                       # x: [B, 168, 1]
        out, _ = self.encoder(x)
        return self.head(out[:, -1])            # [B, 24]


def to_tensors(windows, mu, sd):
    X = np.stack([(h - mu) / sd for h, _, _ in windows]).astype(np.float32)[..., None]
    Y = np.stack([(y - mu) / sd for _, y, _ in windows]).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y)


def train_one(cell, seed, data, mu, sd):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Xva, Yva = data
    model = RecurrentForecaster(cell).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)

    best, best_state, wait = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva.to(DEVICE)).cpu()
        vmae = (vpred * sd - Yva * sd).abs().mean().item()   # kW space
        if vmae < best - 1e-4:
            best, best_state, wait = vmae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def main():
    s = pd.read_parquet(ROOT / CFG["data"]["processed_file"])["load_kw"]
    values = s.to_numpy(np.float32)
    splits, meta = build_splits()
    mu = float(values[:meta["train_end"]].mean())
    sd = float(values[:meta["train_end"]].std())

    Xtr, Ytr = to_tensors(splits["train"], mu, sd)
    Xva, Yva = to_tensors(splits["valid"], mu, sd)
    Xte, _ = to_tensors(splits["test"], mu, sd)
    te_true = [y for _, y, _ in splits["test"]]
    data = (Xtr, Ytr, Xva, Yva)

    for cell in ("RNN", "LSTM", "GRU"):
        maes, rmses, smapes, lats = [], [], [], []
        for seed in SEEDS:
            model = train_one(cell, seed, data, mu, sd)
            model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = (model(Xte.to(DEVICE)).cpu().numpy() * sd + mu)   # back to kW
            lat_ms = (time.perf_counter() - t0) / len(Xte) * 1000
            m = aggregate(te_true, [pred[i] for i in range(len(pred))])
            maes.append(m["MAE_kw"]); rmses.append(m["RMSE_kw"])
            smapes.append(m["sMAPE_pct"]); lats.append(lat_ms)
        log_result({
            "model": cell, "split": "test",
            "MAE_kw": round(np.mean(maes), 4), "RMSE_kw": round(np.mean(rmses), 4),
            "sMAPE_pct": round(np.mean(smapes), 4),
            "p50_latency_ms": round(np.median(lats), 3),
            "notes": f"{len(SEEDS)} seeds; MAE std={np.std(maes):.3f}",
        })
        print(f"[rnn] {cell}: MAE {np.mean(maes):.3f}+/-{np.std(maes):.3f}  "
              f"RMSE {np.mean(rmses):.3f}  sMAPE {np.mean(smapes):.1f}")


if __name__ == "__main__":
    main()
