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
