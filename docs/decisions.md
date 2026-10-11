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

