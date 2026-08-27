# Workshop outline: replicating DeepLOB and a pre-registered stock holdout

Hossein Tabasi, 2026. This is a replication-and-extension outline.
**We have not reproduced Zhang et al. Table I or Table II.** FI-2010 is
not on disk. Toy smoke is not a replication.

## 1. Why a separate repository

The sibling `deeplob-transformer` is the modelling lab: several
architectures, two splits, an OFI ablation. A modelling lab will be
tempted to move k, width, and seeds until a cell looks close to 84.47
accuracy. A replication repository exists to make that temptation
visible. It stores (i) the public original numbers with their source
URL and fetch date, (ii) a protocol that freezes splits and seeds,
(iii) a DIFF file that forbids invented gaps, and (iv) one extension
that was chosen before seeing FI-2010.

## 2. What Zhang et al. actually reported

The paper (arXiv:1808.03668, IEEE Transactions on Signal Processing
2019) evaluates DeepLOB on FI-2010 in two setups. Setup 1 is an
anchored day-forward scheme; Table I reports k=10, 50, 100. Setup 2
trains on the first seven days and tests on the last three; Table II
reports k=10, 20, 50. DeepLOB Setup 2 k=10: accuracy 84.47, precision
84.00, recall 84.47, F1 83.40. Setup 2 k=20: 74.85 / 74.06 / 74.85 /
72.82. Setup 2 k=50: 80.51 / 80.38 / 80.51 / 80.35. Setup 1 k=10:
78.91 / 78.47 / 78.91 / 77.66. These figures were copied from the
ar5iv HTML of the arXiv PDF on 2026-08-27. They are literature values,
not our runs.

The paper does not report k=1, 2, 3 or 5. `ORIGINAL_NUMBERS.csv` says
NA_not_in_zhang2019_tables for those rows rather than NA_pending or a
guess. That is the main honesty constraint relative to the sibling
repo's k-grid.

The paper also reports LSE results and a transfer experiment on five
unseen LSE names (Table IV). We do not have LSE data and will not
invent those cells.

## 3. Protocol

Data: FI-2010, z-score features as released, 100-event windows, 40
raw book features in the Zhang layout. Splits: Setup 2 as primary
replication target; chronological 6/1/3 as a robustness column that
the paper did not run. Seeds: (0, 1, 2). Metrics: accuracy (to sit
next to Table II) and macro-F1 (because the paper itself says F1 is
the fair number on an unbalanced set). Training recipe: we will not
claim that ADAM with Keras epsilon=1 was matched until the log says
so. Reduced width is disclosed.

`src/protocol.py` implements the splits without shuffling days. Tests
lock the day lists and the seed tuple. Changing the chronological cut
after seeing test F1 is a protocol violation.

## 4. The pre-registered extension

Train on three FI-2010 stocks, test on two (`docs/EXTENSION.md`).
Assignment frozen as train {0,1,2}, test {3,4}. This is the cheapest
out-of-name check on the public file. It is not Zhang et al. Table IV
(different market). We chose this extension over "microstructure
features vs CNN" before any empirical peek. If stock-holdout looks
bad we will still report it.

## 5. Software status

`replication_smoke.py` trains a flattened MLP for one epoch on
Gaussian windows and appends a TOY row to OUR_NUMBERS.csv. That row's
model field is `mlp_flatten_not_deeplob`. It exists so that the CSV
pipeline is tested. It is not DeepLOB. Conceptual interfaces in
`src/interfaces.py` list the sibling model names; numbers will be
read from CSV, not by importing the sibling package, so a broken
install cannot silently write a replication.

## 6. What has not been run

No FI-2010 load. No DeepLOB train. No Setup 1 folds. No k=20 or k=50
on our code. No stock-holdout. No LSE. No comparison cell in DIFF.md
has a number. **NO FULL RESULTS YET.**

## 7. How DIFF will work later

Gap = our Setup 2 accuracy - 84.47 at k=10. A gap of a few points may
be width and optimiser. A gap of ten points is a failed replication
until proven otherwise. We will not "fix" it by reporting only
macro-F1 if accuracy looks worse, or only accuracy if F1 looks worse.
We will not drop seed 2 if it is the bad one. We will not switch to
Setup 1 because it is closer.

## 8. Risks specific to replication

FI-2010 dumps on the internet differ (normalisation, whether labels
are included, whether all 144 features or only 40 are stored). The
loader in the sibling repo returns synthetic data if the layout is
unrecognised; a replicator who forgets to read the `source` column
will publish TOY numbers as Table II. That is why OUR_NUMBERS has a
`label` field that must equal TOY or FI2010, never blank.

Reduced width: the paper reports ~60k parameters. Our smoke MLP has
far fewer; our sibling DeepLOB is also reduced. A successful
replication may require a `full_width: true` config that we have not
trained.

Label threshold `alpha` and the exact Ntakaris formula (mean of future
mids vs current mid) must match the released labels, not our
re-labelling, when the archive includes labels.

## 9. Workshop 12-minute arc

Slide 1: what Table II actually contains (k=10,20,50). Slide 2: what
it does not contain (k=1,2,3,5; our PnL). Slide 3: protocol freeze.
Slide 4: empty OUR vs ORIGINAL grid with TOY in a footnote. Slide 5:
pre-registered 3/2 stock holdout. Closing sentence: a copied PDF cell
is a citation; a toy MLP is a unit test; neither is a replication.

## 10. Relationship to the sibling repos

`deeplob-transformer` produces numbers. `paper-replication-plus`
consumes them and compares to the PDF. `causal-earnings-ml` and
`temporal-gnn-market-risk` are separate questions (DML on CARs;
temporal GNN AUPRC). They should not borrow DeepLOB accuracy as a
motivation slide for PEAD or for crash forecasting.

## 11. Planned tables (empty)

Table R1: ORIGINAL Setup 2 vs OUR Setup 2 at k=10,20,50 — OUR empty.
Table R2: chronological vs Setup 2, our code only — empty.
Table R3: 3/2 stock holdout — empty.
Table R4: seed range — empty.

## 12. Limitations

No LSE. No Keras bit-match. No claim of "universal features" until
Table R3 exists. Public numbers could contain ar5iv HTML artefacts;
if a PDF-only appendix disagrees we will re-copy from the PDF and
note the correction rather than average the two.

## 13. Next actions

1. Obtain FI-2010 legally.
2. Run sibling DeepLOB with `source=fi2010` and seeds 0,1,2.
3. Fill OUR_NUMBERS from those CSVs only.
4. Write DIFF gaps.
5. Run the frozen 3/2 holdout.
6. Do not add a second extension.

## 14. Implementation map

REPLICATION.md protocol. ORIGINAL_NUMBERS.csv literature.
OUR_NUMBERS.csv ours (TOY or later FI2010). DIFF.md rules.
docs/EXTENSION.md holdout. src/protocol.py seeds and splits.
src/replication_smoke.py toy row. tests/test_protocol.py locks the
protocol. src/interfaces.py name alignment with the sibling.

## 15. Conclusion of the outline

Replication is a protocol plus a refusal to invent cells the original
paper never printed. This repository has the protocol, the public
cells, and a TOY smoke path. It does not have a reproduced Table II.
Until it does, the correct public sentence is that DeepLOB's Setup 2
k=10 accuracy of 84.47 percent is Zhang et al.'s number, fetched from
arXiv, and not ours.


## 16. Notes for a CSE admissions / lab reader

This repository is intentionally boring. It does not introduce a new
layer. It introduces a CSV discipline: literature numbers are labelled
as literature, toy numbers as TOY, and missing paper cells as not in
the paper. That discipline is the difference between a junior research
clone of DeepLOB and a Kaggle notebook that prints 99 percent on a
shuffled LOB window.

If a lab wants a first-year project, the 3/2 holdout is sized for a
single GPU weekend *after* FI-2010 is in place, not before. The
wrong first-year project is to re-tune k until Setup 2 k=10 matches
84.47 and then declare a successful replication. Matching a headline
without matching the split and the metric is not replication.

A second reader comment: TransLOB is in the sibling repo, not here, so
that a Transformer number cannot be quietly dropped into OUR_NUMBERS
under the DeepLOB model name. If we later replicate Wallbridge (2020)
it will be a new ORIGINAL file with its own arXiv id, not an extra
column in this one.

The smoke train uses a binary up/down label on a random walk mid, not
the three-class Ntakaris label. That is another reason its accuracy is
not comparable to 84.47, which is a three-class number on a real book.
Mixing class cardinalities is a silent bug we call out here so that
DIFF.md never subtracts them.


CI runs only `tests/test_protocol.py`. It does not download FI-2010 and
it does not compare OUR_NUMBERS to 84.47. A green CI means the split
function still returns days 0-6 / 7 / 8-10 and the seed tuple is still
(0, 1, 2). That is the entire scientific content of the test suite, and
it is enough: protocol drift is how replication pages rot. When a future
commit changes chronological_split to a 5/5 cut, the test should fail
loudly rather than adapt. The same is true of stock_holdout_3_2. Those
two asserts are the pre-registration, encoded as code.


This outline ends without a results table because none has been earned.
