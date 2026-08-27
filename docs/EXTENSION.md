# Pre-registered extension (not after peeking)

Chosen extension: **train on 3 FI-2010 stocks, test on 2**.

## Why this one

Zhang et al. already include a transfer experiment on LSE names that
were not in the training set (their Table IV). FI-2010 has five stocks
and a 7/3 day split that pools names. A stock-holdout on the *same*
public file is the cheapest out-of-name check that does not require
LSE data. The alternative (microstructure features vs CNN) is left for
a second paper.

## Frozen assignment

`stock_holdout_3_2`: train stocks {0,1,2}, test stocks {3,4} in the
FI-2010 file order. We do not choose the split by validation accuracy.
We do not rotate after seeing results. If the file order of names is
ambiguous, we will lock the five ticker codes in an addendum *before*
training, not after.

## Protocol

- Same labels, same k-grid, same seeds (0,1,2).
- Train DeepLOB (reduced width) on the three names, all training days
  of Setup 2 for those names only.
- Test on the two held-out names, test days only.
- Metrics: macro-F1 and accuracy. Compare to the pooled five-name
  Setup 2 run (the replication), not to Zhang et al. Table IV (different
  market).

## What would count as a result

A small gap vs pooled training would be consistent with Zhang et al.'s
"universal features" narrative. A collapse would be consistent with
name-specific overfitting on ten days. Either is publishable. We will
not switch to the microstructure-vs-CNN extension because this one
looked bad.

## Status

Not run. FI-2010 absent. The helper `stock_holdout_3_2` exists so the
assignment cannot silently change.
