"""Conceptual model interfaces matching deeplob-transformer.

This package does not import the sibling repo (they are separate
distributions). Names and call signatures are kept in sync by hand:

- MLPClassifier(seq_len, n_features) -> logits (B, 3)
- LSTMClassifier(n_features) -> logits (B, 3)
- DeepLOB() -> logits (B, 3) on (B, T, 40)
- ControlledTransformer(n_features) -> logits (B, 3)

When FI-2010 numbers exist they are read from
../deeplob-transformer/results/tables/ as CSV, not via a Python import,
so a broken sibling install cannot fabricate a replication.
"""
MODEL_NAMES = ("mlp", "lstm", "deeplob", "transformer", "ofi", "both")
INPUT_SHAPE = (None, 100, 40)  # batch, T, features
N_CLASSES = 3
