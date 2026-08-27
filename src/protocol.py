"""Replication protocol helpers: fixed seeds and chronological splits."""
from __future__ import annotations
import random
import numpy as np

SEEDS = (0, 1, 2)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def chronological_split(n_days: int = 10, train_days: int = 6,
                        val_days: int = 1) -> dict:
    """Days 1-6 train, 7 val, 8-10 test (0-based indices). No shuffle."""
    tr = list(range(train_days))
    va = list(range(train_days, train_days + val_days))
    te = list(range(train_days + val_days, n_days))
    return {"train_days": tr, "val_days": va, "test_days": te}

def paper_setup2_split(n_days: int = 10, n_train: int = 7) -> dict:
    return {"train_days": list(range(n_train)), "val_days": [],
            "test_days": list(range(n_train, n_days))}

def stock_holdout_3_2(stock_ids=(0, 1, 2, 3, 4)):
    """Pre-registered FI-2010 extension: train on 3 names, test on 2.

    Frozen assignment: train {0,1,2}, test {3,4}. Not chosen after peeking.
    FI-2010 has five stocks; this is the extension in docs/EXTENSION.md.
    """
    ids = list(stock_ids)
    return {"train_stocks": ids[:3], "test_stocks": ids[3:]}
