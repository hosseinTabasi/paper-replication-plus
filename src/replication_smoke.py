"""One-epoch synthetic smoke. Writes a TOY row to OUR_NUMBERS.csv.

This is not a DeepLOB FI-2010 replication. Full tables require FI-2010
and will later be consumed from the sibling repo deeplob-transformer.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from protocol import set_seed, chronological_split, paper_setup2_split

def _toy_windows(n_days=10, n_events=80, seq=16, seed=0):
    rng = np.random.default_rng(seed)
    X, y, days = [], [], []
    for d in range(n_days):
        book = rng.normal(size=(n_events, 40)).astype(np.float32)
        mid = 100 + np.cumsum(rng.normal(0, 0.02, n_events))
        for t in range(seq, n_events - 1):
            X.append(book[t-seq:t])
            y.append(int(mid[t+1] > mid[t]))
            days.append(d)
    return np.stack(X), np.asarray(y), np.asarray(days)

def smoke(root: Path | None = None, seed: int = 0) -> dict:
    set_seed(seed)
    root = Path(root) if root else Path.cwd()
    X, y, days = _toy_windows(seed=seed)
    split = paper_setup2_split()
    tr = np.isin(days, split["train_days"])
    te = np.isin(days, split["test_days"])
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        model = nn.Sequential(nn.Flatten(), nn.Linear(16 * 40, 32), nn.ReLU(), nn.Linear(32, 2))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                            batch_size=32, shuffle=True)
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X[te])).argmax(-1).numpy()
        acc = float((pred == y[te]).mean())
        backend = "torch_mlp_toy"
    except Exception as exc:
        acc = float("nan"); backend = f"skipped:{type(exc).__name__}"
    row = {
        "label": "TOY",
        "source": "synthetic_smoke",
        "model": "mlp_flatten_not_deeplob",
        "k": 1,
        "setup": "paper_setup2_on_toy_days",
        "seed": seed,
        "accuracy": None if acc != acc else round(acc, 6),
        "macro_f1": "NA_not_computed",
        "note": "TOY 1-epoch MLP; NOT a DeepLOB FI-2010 replication",
        "backend": backend,
    }
    out = root / "OUR_NUMBERS.csv"
    fields = list(row.keys())
    exists = out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)
    return row

if __name__ == "__main__":
    print(smoke())
