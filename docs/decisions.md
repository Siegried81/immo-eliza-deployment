# Decisions

One dated entry per change that alters what a number means. Append, never rewrite.

## 2026-10-10: The state dictionaries are keyed the way the data spells the states

- **What:** `ENERGY_MEAN_BY_STATE` and `STATE_MAPPING` in `src/features.py` are
  keyed `"TO RENOVATE"`, `"FULLY RENOVATED"`, … (upper case, spaces), which is
  how `models/categories.json` and the training rows spell them. Input with an
  underscore or stray whitespace is folded into that spelling. `streamlit/app.py`
  no longer rewrites spaces to underscores before the lookup.
- **Why:** the keys carried underscores, which no row ever matched, so every
  multi-word state fell through to `NORMAL`: ordinal 3 instead of 2 for
  "TO RENOVATE", and 130 kWh instead of 180 where energy was missing. That
  happened in training and at inference alike, so the models never saw the
  distinction the dictionaries were meant to encode.
- **What changes in the numbers:** `property_state_encoded` and the imputed
  energy change for five of the eight states. The pickled models were fitted on
  the old values, so they must be retrained before this ships:
  `python src/train.py`, `python scripts/export_categories.py`, then the
  README's metrics table re-measured with `python monitoring/metrics.py`.
- **Revisit if:** a state appears in the data that is in neither dictionary
  (`tests/test_predict.py::TestStateEncoding` fails on it).

## 2026-10-11: the routing table follows the retrained models

`scripts/evaluate_stratified.py` was re-run on the validation split (3,150 rows)
with the models retrained after the state-key fix and committed in `bc8e6b1`.
Every cell of the "Luxury Routing" table moved by a few tenths; the choice of
0.9 holds for the same reason as before (the luxury tier falls from 46% to 35%
MAPE while the < €1M tier moves 21.9% → 22.3%). One thing the new run states
that the old table did not: against no routing, 0.9 costs 0.5 point of overall
MAPE (22.3% → 22.8%). Routing is kept for the top tier, not for the average,
and the README now says so. The routed counts are 83 / 40 / 23 at 0.7 / 0.9 /
0.95 (were 80 / 42 / 24).

## 2026-10-11: a city typed in the other national language reaches the model

The training data keeps one spelling per Belgian municipality (mostly the Dutch
name in Flanders and Brussels, the French one in Wallonia, a few English ones).
A user typing "Ixelles", "Luik" or "Gand" got "city not in the training data" and
an estimate without the city, for cities the models know. `CITY_ALIASES` maps 80
French/Dutch/English names to the spelling in `models/categories.json`; the
user's province is still preferred when a target name exists in two provinces.

Taken from the branch `merge/app-overhaul-into-main`, which fixed the state keys
the other way (underscores) and could not be merged over the space-keyed fix
already on main. Only the alias table, its lookup and its tests were ported.
No model, metric or encoding changed. Tests: 36 -> 57 passed (the README said
31; re-measured).


## 2026-10-12: README accuracy figures name the pipeline they measure

The "Known Limitations" accuracy line quoted the standard model alone
(€68k below €1M, +5% overall bias, `monitoring/metrics.py`) while the UI
caption quoted the deployed pipeline with routing at 0.9 (€70k, +7% below
€1M, `scripts/evaluate_stratified.py`). Both scripts were re-run on 2026-10-12
on the committed models, without retraining, and every published cell still
matched. The limitations line and the conclusion now quote the deployed
pipeline (22.8% overall, 22.3% and +6.6% below €1M, 34.9% above €1.9M); the
conclusion said 22.9% / 22% / 34%. The bug #10 row now quotes the coverage in
`models/interval_calibration.json` (70.7% -> 77.8%), as the interval paragraph
already did; it said 70.4% -> 77.6%. No model or metric changed.

## 2026-10-12: the drift report measures logged features against the training split

- **What:** the prediction log is JSON Lines (`monitoring/logs.jsonl`, or
  `PREDICTION_LOG_PATH`), one line appended per prediction, carrying the
  engineered features the models read and a `source` ("api" or "synthetic").
  `check_drift.py` compares those features with `train.py`'s training split
  (12,596 rows), not `data/training_baseline.csv` (10,802 rows, not the
  training split). `generate_logs.py` writes validation rows to its own file
  and never deletes the real log.
- **Why:** the API logged raw input without `property_state_encoded`, so a
  report on real traffic raised KeyError (reproduced on the previous code with
  60 API requests); the generator deleted the real log before writing; and the
  README's table compared random values, including a state the model does not
  know, with a baseline that was not the training data.
- **What changes in the numbers:** the README's PSI table. It said 7 of 12
  features drifted strongly; on 2,000 validation rows the tooling now reads
  every feature as stable (PSI 0.0000-0.0146). Neither table says anything
  about production, and the README says so. No model, metric or encoding
  changed.

## 2026-10-12: "deterministic" means on one machine

Retraining in a Linux container (xgboost 3.2.0, scikit-learn 1.9.0) twice, and
with the previous and the refactored `train.py`, gave identical models each
time. Against the committed models, the quantile models, the routing
classifier, `luxury_threshold.json` and `interval_calibration.json` match
exactly, but the standard and luxury regressors differ by up to 0.38 and 0.40
in log price on the validation rows. The README said "two runs give identical
models" without that condition; it now says "on a given machine". The
committed models were not replaced: the published figures are theirs.

## 2026-10-12: the clip bounds and the luxury price come from the training rows

- **What:** `src/train.py::price_bounds()` takes the 1st and 99th percentiles of
  the TRAINING prices; they set the clip of the standard and quantile models'
  target and the luxury price (and so the routing labels and the luxury tier).
  They were taken on all rows, validation included.
- **Why:** the validation prices helped decide what the models were trained on
  and how the luxury tier was cut, so the scores were not entirely out of
  sample. Measured: the 99th percentile moves from €1,900,000 to €1,890,000 and
  the 1st from €68,500 to €68,475.
- **What changes in the numbers:** all five models were retrained (in the
  python:3.12 Linux image, the deploy image) and every README and deep-dive
  figure re-measured. Standard model: MAPE 22.34% -> 22.52%, bias +4.62% ->
  +5.06%. As deployed (routing 0.9): 22.8% -> 23.0% overall, 22.3% -> 22.4%
  below €1M, 34.9% -> 35.8% on the luxury tier, which now holds 36 rows (was
  35). Retraining the PREVIOUS code in the same image gives 22.64% and 23.2%:
  the threshold fix itself improves both figures; the rest of the move is the
  machine (see the 2026-10-12 entry on determinism). The 0.9 routing threshold
  still holds for the same reason. Range coverage on the check half: 77.9%,
  80.1% on average over 200 halvings.

## 2026-11-08: a left-out bedroom count is missing, not 1

- **What:** `api/app.py::PropertyInput.bedroom_count` defaults to `None`
  instead of `1`. A request without it now reaches the pipeline as NaN and is
  filled by the median imputer fitted in `src/train.py` (3 bedrooms).
- **Why:** the schema promises that optional fields left out are imputed as in
  training. Bedrooms were the exception: an API or CLI caller who skipped the
  field got a one-bedroom price. Measured on the committed models, a 150 m²
  Liège house went from €251,749 to €212,429 (-15.6%) and an 85 m² Elsene
  apartment from €343,417 to €319,310 (-7.0%).
- **What changes in the numbers:** only the price of requests that leave
  `bedroom_count` out. A request that sends it, and every Streamlit request
  (the UI always sends it), gets the same price as before. No model, metric or
  README performance figure changes; no retraining.
- **Revisit if:** the imputer strategy in `build_preprocessor()` changes.

## 2026-11-08: a column the caller leaves out is missing in add_features

- **What:** `src/features.py::add_features()` creates an absent
  `bedroom_count`, `preschool_distance_m`, `train_station_distance_m` or
  `supermarket_distance_m` column as missing (None) instead of 1 or 0. The
  pipeline's median imputer then fills it.
- **Why:** a distance of 0 m and one bedroom are real values, far from the
  training medians (502 m, 2,000 m, 706 m, 3 bedrooms). The API always sends
  these keys, so it never hit the defaults, but a direct `engine.predict()`
  with the required fields alone did, and the Hugging Face model card shows
  exactly that call. Measured on a 90 m² Ixelles apartment built in 2000: engine
  €300,692, API €363,063 (-17.2%). After the fix both give €363,063.
- **What changes in the numbers:** only direct engine calls that leave these
  fields out. The training data has every column, so training, the committed
  models and every published metric are unchanged; no retraining.
- **Still open:** `build_year`. Left out, the API uses 2000, while
  `add_features` fills a missing year with `CURRENT_YEAR` (age 0), as it did
  for training rows. Which one is right is a modelling decision, not a typo.

## 2026-11-10: a sealed test split, and one rule for a missing build year

- **What:** `src/train.py::split_all()` cuts the 15,746 listings into four
  disjoint splits with `SPLIT_SEED = 42`: train 11,022 (70%, fits everything),
  validation 1,575 (10%, early stopping, the Optuna search, checking the
  routing threshold), calibration 787 (5%, the range's conformal margin,
  nothing else) and test 2,362 (15%, sealed). Only the final figures read the
  test rows: `monitoring/metrics.py`, `scripts/evaluate_stratified.py --test`
  and the coverage in `models/interval_calibration.json`. `split()` still
  returns (train, validation) for the scripts that need the training rows.
- **Why:** every published figure came from the old 3,150-row validation split,
  and that split also early-stopped the standard and quantile models, scored
  the Optuna search and picked the 0.9 threshold. So "22.5% MAPE on 3,150
  unseen listings" was a validation score, optimistic, not a test score. The
  range had the same problem: its margin was set on half of those rows, which
  the quantile models had early-stopped on, so the conformal guarantee didn't
  strictly hold.
- **Where the test rows come from:** the old 80% training split only. A fresh
  15% drawn with the same seed would have been 2,362 of the old 3,150
  validation rows, every one of them seen by the October threshold choice and
  search. Drawn from the old 80%, no earlier choice was ever scored on them.
- **Also fixed with this retraining:**
  - A missing `build_year` stays missing in `add_features()` and at the API
    (`PropertyInput.build_year` defaults to None). The median imputer fills
    it with the training median, 1974. Before, training used the current
    year (a new build) and the API sent 2000. A 150 m² Liège house without a
    year is now €220,213.
  - The classifier's `scale_pos_weight` is counted on the training rows only
    (≈ 98). It was counted on all rows.
  - `src/optimize.py` is seeded and was re-run on the new train and
    validation splits (50 trials); its best parameters are in `train.py`.
  - `monitoring/generate_logs.py` writes 1,500 validation rows by default,
    since the split now has 1,575.
- **What changes in the numbers:** all five models retrained in the
  python:3.12 Linux image (pinned versions). The clip bounds and luxury price
  moved with the training rows: €69,000 to €1,948,950 (was €68,475 to
  €1,890,000), 111 luxury training rows (was 125).

  | Figure | Before (old validation, 3,150 rows) | After (test, 2,362 rows) |
  |---|---:|---:|
  | Standard model MAPE | 22.52% | 26.24% (22.17% without row 5258) |
  | Standard model median APE | not published | 14.85% |
  | Standard model MAE | €92,685 | €92,024 |
  | Standard model R² (euros) | 0.659 | 0.646 |
  | Standard model bias | +5.06% | +7.88% (+3.79% without row 5258) |
  | As deployed (0.9), overall MAPE | 23.0% | 26.5% (22.5% without it) |
  | As deployed, R² (euros) | 0.663 | 0.617 |
  | As deployed, < €1M MAPE | 22.4% | 26.0% (21.8% without it) |
  | As deployed, < €1M bias | +7.1% | +9.9% (+5.6% without it) |
  | As deployed, top tier MAPE | 35.8% (36 rows) | 46.2% (21 rows) |
  | As deployed, top tier bias | −29% | −43.0% |
  | Range coverage, raw to calibrated | 70.7% → 77.9% (1,555 rows) | 73.6% → 80.9% (2,344 rows) |
  | Range median width | ≈ €213k | ≈ €213k |

  Row 5258 is a 77 m² Oostende apartment listed at €3,048, a data error. The
  model says €297,380, a 9,657% error, which alone adds 4 points to the MAPE
  of 2,362 rows. It stays in the test split: dropping it after seeing the
  score would be a choice made on the test. The cleaning should drop prices
  like that; once it does, the test split has to be drawn again.

  The top tier got worse (35.8% to 46.2%). The figures can't tell how much is
  the old split flattering the old models and how much is noise: 21 rows, and
  one property moves the tier's MAPE by about 5 points.

  No README in this repo ever quoted an R² of 0.79. That figure comes from
  `immoeliza_ml` (0.7956 for `models/XGBoost.py`, 0.7908 for the tuned
  script), a different model on a different preprocessing.

  77.8% or 77.9%: the 2026-10-12 entry on README figures quotes 77.8%, which
  was right for the models committed at that time (0.7783). The retraining
  later that day gave 0.7788, so 77.9% was the right figure for the models
  replaced today.
- **The threshold:** 0.9 is kept and checked, not re-tuned: the new
  validation split has 19 rows above €1.95M. There, 0.9 cuts the top tier's
  MAPE from 51.2% to 44.0% and costs 0.3 point below €1M. On the test split
  it does the same (51.6% to 46.2% on the top tier, 0.3 point overall).
- **Revisit if:** a threshold or any other setting gets chosen by looking at
  `evaluate_stratified.py --test`. Then the test split is spent and needs a
  new draw or new data.
