# Replication protocol (DeepLOB, Zhang et al. 2019)

This document freezes how a full replication *will* be run. It is not a
claim that the full tables have been reproduced. FI-2010 is not on disk.

## Data

- Dataset: FI-2010 (Ntakaris et al. 2018), five Nasdaq Nordic stocks,
  ten days, 10 LOB levels, 40 features. See sibling repo
  `deeplob-transformer/scripts/download_fi2010.md`.
- Do not scrape. Do not commit the archive.

## Splits

- **Paper Setup 2** (deep-learning convention in Zhang et al. Table II):
  first 7 days train, last 3 days test. No validation day in the paper.
- **Chronological robustness** (this project, not the paper): days 1-6
  train, day 7 validation, days 8-10 test. Implemented in
  `src/protocol.py:chronological_split`. Seeds never shuffle days.

## Seeds

Frozen tuple `SEEDS = (0, 1, 2)` in `src/protocol.py`. Report mean and
range across seeds. Do not add a fourth seed after seeing test macro-F1.

## Metrics

Zhang et al. headline accuracy, precision, recall, F1 (they note FI-2010
is unbalanced and F1 is the fair comparison). We will additionally
report **macro-F1**. We will not report PnL.

## Horizons

The paper uses k=10, 20, 50 on Setup 2 (and k=10, 50, 100 on Setup 1).
It does **not** report k=1,2,3,5. `ORIGINAL_NUMBERS.csv` records that
fact instead of inventing cells. Our research grid in the sibling repo
includes k=1,2,3,5,10; overlap with the paper is k=10 (and we will also
run k=20,50 when FI-2010 exists).

## Model

Reduced-width DeepLOB in `deeplob-transformer/src/models/deeplob.py`,
not a GPL dump. A full-width match to the ~60k-parameter paper network
is a later config, not the smoke.

## What counts as a successful replication

A cell-by-cell comparison in DIFF.md after an FI-2010 run, with
absolute accuracy gap and a written discussion of width, optimiser,
and label threshold. A TOY MLP accuracy is not a successful
replication. This file will be edited to "ran" only after FI-2010.
