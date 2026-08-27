"""Seeds are fixed; chronological split is ordered; no day leakage."""
from __future__ import annotations
from protocol import SEEDS, chronological_split, paper_setup2_split, set_seed, stock_holdout_3_2
import numpy as np

def test_seeds_frozen():
    assert SEEDS == (0, 1, 2)
    set_seed(0)
    a = np.random.rand(4)
    set_seed(0)
    b = np.random.rand(4)
    assert np.allclose(a, b)

def test_chronological_ordered_and_disjoint():
    s = chronological_split(n_days=10)
    assert s["train_days"] == [0, 1, 2, 3, 4, 5]
    assert s["val_days"] == [6]
    assert s["test_days"] == [7, 8, 9]
    assert set(s["train_days"]).isdisjoint(s["test_days"])
    assert set(s["train_days"]).isdisjoint(s["val_days"])
    # strictly increasing (no shuffle)
    assert s["train_days"] == sorted(s["train_days"])

def test_paper_setup2():
    s = paper_setup2_split()
    assert s["train_days"] == list(range(7))
    assert s["test_days"] == [7, 8, 9]

def test_extension_3_2_frozen():
    h = stock_holdout_3_2()
    assert h["train_stocks"] == [0, 1, 2]
    assert h["test_stocks"] == [3, 4]
