# Replication + one extension of DeepLOB

**Author:** Hossein Tabasi (2026). MIT licence. Replication notebook-as-package.

## Question

Can we reproduce Zhang, Zohren and Roberts (2019) DeepLOB mid-price
classification tables on FI-2010 under a frozen protocol (data, splits,
seeds, metrics), and does a *pre-registered* train-on-3 / test-on-2
stock holdout still show transferable book features?

## Why it matters

LOB papers are hard to compare because k, splits, and F1 vs accuracy
move together. A replication file that copies public table cells and
refuses to invent the missing k=1,2,3,5 rows is a prerequisite for
claiming that a later Transformer (sibling repo) improved on DeepLOB.
The extension is registered in `docs/EXTENSION.md` before FI-2010 is
seen.

## Data

FI-2010, same as the sibling `deeplob-transformer`. Not shipped. Toy
smoke uses Gaussian windows labelled TOY.

## Method

Protocol in `REPLICATION.md` and `src/protocol.py`. Setup 2 (7/3 days)
and a stricter chronological split. Seeds (0,1,2). Public original
cells in `ORIGINAL_NUMBERS.csv` from arXiv:1808.03668 Table I/II
(fetched via ar5iv HTML on 2026-08-27). Our cells in `OUR_NUMBERS.csv`.
Comparison rules in `DIFF.md`.

## Baselines

The original DeepLOB numbers are the literature baseline. Our reduced-
width DeepLOB (sibling) is the replication attempt. A flattened MLP
smoke in this repo is only an integration check.

## Results

**NO FULL RESULTS YET.** We have **not** reproduced DeepLOB full tables
because we have not run FI-2010. `ORIGINAL_NUMBERS.csv` contains
literature values (and NA where Zhang et al. did not report k=1,2,3,5).
`OUR_NUMBERS.csv` contains at most a TOY smoke row. Do not cite TOY
accuracy as a replication.

## Limitations

Reduced width. No LSE data (the paper's second dataset). No PnL.
k-grid mismatch with the paper is documented, not papered over.

## Reproduce (toy)

PYTHONPATH=src python -m replication_smoke
PYTHONPATH=src python -m pytest -q

## References

Zhang, Zohren, Roberts, IEEE TSP 2019, arXiv:1808.03668.
Ntakaris et al., Journal of Forecasting 2018 (FI-2010).
