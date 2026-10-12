# Decisions

One dated entry per change that alters what a number means. Append, never rewrite.

## 2026-10-10 — The state dictionaries are keyed the way the data spells the states

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
- **Revisit if:** a state appears in the data that is in neither dictionary —
  `tests/test_predict.py::TestStateEncoding` fails on it.

## 2026-10-11 — the routing table follows the retrained models

`scripts/evaluate_stratified.py` was re-run on the validation split (3,150 rows)
with the models retrained after the state-key fix and committed in `bc8e6b1`.
Every cell of the "Luxury Routing" table moved by a few tenths; the choice of
0.9 holds for the same reason as before (the luxury tier falls from 46% to 35%
MAPE while the < €1M tier moves 21.9% → 22.3%). One thing the new run states
that the old table did not: against no routing, 0.9 costs 0.5 point of overall
MAPE (22.3% → 22.8%) — routing is kept for the top tier, not for the average,
and the README now says so. The routed counts are 83 / 40 / 23 at 0.7 / 0.9 /
0.95 (were 80 / 42 / 24).

## 2026-10-11 — a city typed in the other national language reaches the model

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


## 2026-10-12 — README accuracy figures name the pipeline they measure

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

## 2026-10-12 — the drift report measures logged features against the training split

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

## 2026-10-12 — "deterministic" means on one machine

Retraining in a Linux container (xgboost 3.2.0, scikit-learn 1.9.0) twice, and
with the previous and the refactored `train.py`, gave identical models each
time. Against the committed models, the quantile models, the routing
classifier, `luxury_threshold.json` and `interval_calibration.json` match
exactly, but the standard and luxury regressors differ by up to 0.38 and 0.40
in log price on the validation rows. The README said "two runs give identical
models" without that condition; it now says "on a given machine". The
committed models were not replaced: the published figures are theirs.
