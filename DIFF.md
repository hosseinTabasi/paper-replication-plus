# How to compare ORIGINAL vs OUR numbers

## Honest gaps (current)

- ORIGINAL_NUMBERS.csv holds Zhang et al. (2019) **public** Setup 1/2
  cells copied from arXiv HTML (ar5iv) on 2026-08-27. k=1,2,3,5 are
  NA_not_in_zhang2019_tables because those horizons are not in Table I/II.
- OUR_NUMBERS.csv holds at most a **TOY** one-epoch MLP row from
  `src/replication_smoke.py`. That row is not DeepLOB and not FI-2010.
- Therefore there is **no numerical comparison to report**. Filling a
  gap column with invented percentages is forbidden.

## When FI-2010 has been run

1. Pull rows from `deeplob-transformer/results/tables/` (source=fi2010).
2. Align on (model=DeepLOB, setup, k).
3. Gap = our_accuracy - original_accuracy. Report sign and magnitude.
4. Expected sources of gap: reduced width, different optimiser
   (paper uses ADAM lr=0.01, epsilon=1, batch 32, Keras), label
   threshold `alpha`, z-score already applied in FI-2010, seeds.
5. A gap larger than 5 percentage points at k=10 Setup 2 is a
   replication failure until diagnosed, not a new SOTA claim.

## What DIFF is not

DIFF is not a leaderboard. It is not a licence to drop the chronological
split if it looks worse. It is not a trading comparison.
