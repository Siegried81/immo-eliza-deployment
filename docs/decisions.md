# Decisions

One entry per decision that still holds: date, what, why. When a later change replaces an earlier one, the entry carries the latest date and names the earlier one in a few words. Oldest first.

## 2026-10-10: state dictionaries are keyed the way the data spells states

`ENERGY_MEAN_BY_STATE` and `STATE_MAPPING` in `src/features.py` use `"TO RENOVATE"`, `"FULLY RENOVATED"` (upper case, spaces), as in `models/categories.json`. Input with an underscore or stray whitespace is folded into that spelling.
Why: the old keys had underscores, matched no row, and sent every multi-word state to `NORMAL`, in training and at inference alike. Five of the eight states changed encoding, so all models were retrained.
Revisit if a state appears in the data that is in neither dictionary (`tests/test_predict.py::TestStateEncoding` fails on it).

## 2026-10-11: a city typed in the other national language reaches the model

The training data keeps one spelling per municipality. "Ixelles", "Luik" or "Gand" used to get "city not in the training data" and an estimate without the city. `CITY_ALIASES` maps 80 French, Dutch and English names to the training spelling; the user's province wins when a name exists in two provinces. No model or metric changed.

## 2026-10-12: "deterministic" means on one machine

Two retrains in the same Linux image give identical models. Against models retrained on another machine, the standard and luxury regressors differ by up to 0.40 in log price (XGBoost depends on CPU and thread count). So part of any metric move after a retrain is the machine. The committed models are trained in the `python:3.12` image with the pinned versions, and the docs say "on a given machine".

## 2026-10-12: the drift report measures logged features against the training split

The prediction log is JSON Lines (`monitoring/logs.jsonl`, or `PREDICTION_LOG_PATH`): one line per prediction with the engineered features the models read and a `source` ("api" or "synthetic"). `check_drift.py` compares them with `train.py`'s training split, not with `data/training_baseline.csv`, which is not the training split. `generate_logs.py` writes to its own file and never deletes the real log.
Why: the API used to log raw input, so a report on real traffic raised KeyError, and the README's PSI table compared random values with the wrong baseline. On 2,000 validation rows every feature reads as stable. That says nothing about production.

## 2026-10-12: clip bounds and the luxury price come from the training rows

`src/train.py::price_bounds()` takes the 1st and 99th percentiles of the training prices only. They clip the standard and quantile targets and define the luxury price, so they also define the routing labels and the luxury tier. Taken on all rows, validation prices helped decide what the models trained on, and the scores were not fully out of sample.
Current values: €69,000 to €1,949,050, 111 luxury training rows.

## 2026-11-10: a sealed test split

`src/train.py::split_all()` cuts the listings into four disjoint splits with `SPLIT_SEED = 42`: train 70% (fits everything), validation 10% (early stopping, the Optuna search, the routing threshold), calibration 5% (the range margin, nothing else), test 15% (sealed). Only the final figures read the test rows: `monitoring/metrics.py`, `scripts/evaluate_stratified.py --test` and the coverage in `models/interval_calibration.json`.
Why: every earlier figure came from a split that also tuned the models, so "22.5% MAPE on unseen listings" was a validation score. The test rows were drawn from the old 80% training part, so no earlier choice was ever scored on them.
Also from this retrain: `scale_pos_weight` is counted on training rows only (about 98); `src/optimize.py` is seeded; the 0.9 routing threshold is kept and checked, not re-tuned (it cuts the top tier's MAPE and costs 0.3 point overall).
Revisit if a threshold or any setting gets chosen by reading the test split. Then the split is spent and needs new data or a new draw.

## 2026-11-15: an optional field left out is missing, not a default

`bedroom_count`, `build_year`, `terrace` and the three distances default to missing in the API schema and in `add_features()` (`livable_surface` is required by the API). The median imputer fitted in training fills them (3 bedrooms, build year 1974, terrace 1, distances 502 m, 2,000 m and 707 m). Garage and pool keep 0: no training row lacks them.
Why: a default of 0 or 1 is a real value far from the medians. Measured on the models of the time: a 150 m² Liège house without bedrooms went from €251,749 to €212,429 (-15.6%); a direct `engine.predict()` without distances gave 17.2% less than the API; a missing terrace came out up to 13% lower through the API. Steps: bedrooms and distances on 2026-11-08, build year on 2026-11-10 (it was 2000 in the API and the current year in training), terrace on 2026-11-15.
Check: replaying 364 test listings through `PropertyInput` and `engine.predict()` matched the scored pipeline on all of them.

## 2026-11-15: the models read the real energy consumption

`to_model_columns()` renames the data's `energy_consumption_kWh/m2/year` to the name the code reads and raises if a model input has no column. A consumption of 0 or above 1,000 kWh (`MAX_ENERGY_KWH_M2_YEAR`) counts as not given and gets the figure for the property's state.
Why: the two names never matched, so the column was filled from the state for every row and no model read a consumption (12,841 of 15,746 listings give one). At inference a real 250 kWh cost 16%. 218 values are above 1,000, up to 20,260,607, which is a date.
`ENERGY_MEAN_BY_STATE` is rebuilt from the data: medians rounded to 10 (70 new, 180 excellent, 220 fully renovated, 260 normal, 420 to renovate; 470 and 600 for "to restore" and "to demolish", which have 10 listings and 1 and only keep the order). With the old 130 kWh for a normal state, a consumption left out priced the house like a frugal one.
Cross-validation below €1M: MAPE 22.08% with the column unread, 21.26% read and capped. The table change alone is neutral (21.26% against 21.27%): it changes what "unknown" means, not the accuracy.
Revisit if the training data changes: the table is computed from it by hand.

## 2026-11-15: the models read the postcode

`postcode`, `area_latitude` and `area_longitude` are model inputs. The coordinates are the median of the training listings in the postcode, else in the city, else missing (`src/features.py::add_location`). `src/train.py` writes them to `models/locations.json`, filed by province because nine city names exist in two provinces. The file also holds each city's most common postcode, which the form starts on.
Why: `postcode` was required by the API and read by nothing. A request carries no coordinates, and in cross-validation the postcode centroid does as well as exact coordinates (20.68% below €1M for both, 21.66% without location). It also makes a postcode the same place in training and at inference.
Every price changes: the same Liège house is €241,607 in postcode 4000 and €229,781 in 4020. About 1% of test listings carry a postcode the training rows lack; they get the city's centroid and the response says so.

## 2026-11-15: asking prices under €10,000 are dropped, from every split

`MIN_PLAUSIBLE_PRICE = 10_000` in `src/train.py`; `split_all()` drops those rows after drawing the splits. Three rows go (a house at €2,500, a house at €5,000, and a 77 m² Oostende apartment at €3,048); the next price up is €19,900.
Why after the draw: every other row keeps its split, so no row used for tuning can move into the test split. The rule reads the price alone, never a model's error. It was written after seeing the €3,048 row in the test split, and the bound comes from the price distribution, not from the score.
Revisit if the upstream cleaning drops these rows itself. Then the constant does nothing and can go.

## 2026-11-15: what the published figures mean

The figures are test-split figures, 2,361 rows: standard model MAPE 20.28%, median 12.88%, MAE €85,137, R² 0.676; as deployed with routing at 0.9, 20.6% overall, 19.9% below €1M, 44.0% on the top tier (20.3%, 19.8% and 43.9% since the luxury cap of 2026-11-24); range coverage 70.8% raw, 80.4% calibrated. `monitoring/metrics.py` scores the standard model alone, `scripts/evaluate_stratified.py --test` scores what the API serves. Say which one a figure is.
Of the 5.9 points of overall MAPE gained since 2026-11-10, 4 are the €3,048 row leaving and about 2 are the models.
The test split was read three times that day (energy, postcode and price rule, state table). Each change was chosen on train and validation first, and none was kept or dropped for its test score: the state table reads 0.2 point worse and was kept for what it means.
Measured and not done: leaving missing values to XGBoost, an "unknown" state category or a count of missing fields moved the cross-validated error below €1M by 0.2 point at most.

## 2026-11-15: bias is published as a median and by price band

The UI caption and README no longer say prices below €1M are overestimated by about 6%. That was the mean of (estimate - price) / price, which is positive for a model that is too high as often as too low: twice the price weighs +100%, half the price -50%. Below €1M, as deployed, the mean is +5.1%, the median +0.3%, and the estimates sum to 1.6% under the prices. The median bias is +22.4% under €150k (82% of 257 rows estimated above their price), +4.3% from €150k to €250k, -1.4% from €250k to €500k and -9.8% from €500k to €1M. `evaluate_stratified.py` prints mean, median and total bias per tier; the mean is still published, named "mean bias".

## 2026-11-15: the same building can sit on both sides of the split

155 of the 2,361 test listings share their address with a training listing, and 27 are exact twins. As deployed the error is 10.7% on those and 21.0% on the other 2,206, against 20.3% on all rows (figures with the luxury cap of 2026-11-24). The published figures are right for a listing drawn like the data; for a building the models never saw they are about 0.7 point optimistic, which is inside the noise of the test split (95% CI on MAPE about ±1.3 points). The README says so. A split grouped by address would fix it and needs a new test split, which is a design decision.

## 2026-11-22: audit pass

No model and no published figure changed.
- `luxury_model_available` is read. `src/train.py` wrote the flag and nothing read it, so a run with fewer than 20 luxury rows would have left the old luxury model in place and still routed to it. `api.predict.load_routing_threshold()` now returns an infinite threshold when the flag is false. The committed models (111 luxury rows) behave as before.
- The luxury price prints as €1.95M in the tier labels (it printed €1.9M for €1,949,050). `STRONG_PSI` is defined once, in `monitoring/monitor.py`, and `evaluate_routing_blend.py` takes the deployed threshold from `src/train.py`.
- `garden_m2` is deprecated and ignored: no model reads it. It stays in the schema so existing clients still validate.
- Modules are imported from the repository root and run with `python -m`; pytest and ruff are configured in `pyproject.toml`. Only `streamlit/app.py` edits `sys.path`, because `streamlit run` puts only `streamlit/` on it. `api.app` reads `PORT`.
- Requirements are pinned, with `requirements-dev.txt` for tests and lint. `python-dotenv` is gone: nothing imports it.
- The API image copies only `api/`, `src/`, `monitoring/` and `models/` and runs as a non-root user. Publishing fails on a CRITICAL CVE that has a fix (Trivy).
- The docs were cut to what a reader needs.

## 2026-11-24: the drift report measures drift it used to miss

`monitoring.monitor.psi()` takes its bins from the training deciles, not from ten equal-width bins over both samples. Province, city, type and state are now compared on their shares (`categorical_psi`). `check_drift.py` needs 300 logged predictions, up from 50.
Why: equal-width bins over a long tail put 97% of the training surfaces in one bin. A shift of every surface by x1.5 read 0.076 ("OK"); it now reads 0.377. Doubled plot surfaces went from 0.030 to 0.561. In the other direction, 50 rows drawn from the validation split with no drift at all could read above 0.25 ("STRONG DRIFT"). Over 200 draws, the noisiest feature's 95th percentile PSI is 0.22 at 100 rows, 0.12 at 200 and 0.085 at 300, so from 300 a stable feature stays under 0.1.
Any PSI computed before this date is not comparable with one computed after.

## 2026-11-24: audit pass

No model and no headline figure changed.
- The range is widened to include the point estimate. Three separate models priced 12 of the 2,344 standard-segment test listings outside their own range. Coverage moves from 80.42% to 80.46%; `interval_calibration.json` still records the 80.42% that `train.py` measures without this rule.
- `add_features()` folds garage and terrace to 0/1. The training data only holds 0 and 1, and the API reads any positive value as yes, so the price does not move; the drift log now stays on the training scale.
- The API bounds every field well above the largest training value and caps text at 64 characters. A surface of 10^15 m² or a 100 kB city name used to get a price and a log line.
- Only an unknown category (`src.features.UnknownCategory`) answers 422. Any other error while predicting is a 500 without its message.
- The README gives the MAPE with its bootstrap interval, the share of estimates within 10% and 20%, say the range figure covers the standard segment only, and give R² on both scales.

## 2026-11-24: the luxury price is capped at 1.5 times the standard estimate

A property routed to the luxury model (probability 0.9 or more) gets the luxury estimate, capped at 1.5 times the standard model's (`src/train.py::LUXURY_MAX_RATIO`, `routed_price()`). The API, the CLI and both evaluation scripts call the same function.
Why: the luxury model learned only prices above the luxury price (€1,949,050), so it never answers below about €2.2M. On the test split, 10 of the 17 routed listings were not luxury (€0.99M to €1.75M) and got about twice their price.
Chosen on the validation split, 13 routed rows: no cap, 1.25, 1.5 and 2.0, and a gate on the standard estimate. 1.5 kept the top tier at 40.8%, brought the €1M to €1.95M tier from 28.1% to 26.8% and the overall from 21.07% to 20.98%; 1.25 cost the top tier 4 points, 2.0 changed nothing.
Measured afterwards on the test split, not chosen on it: overall MAPE 20.6% to 20.3%, below €1M 19.9% to 19.8%, €1M to €1.95M 31.1% to 27.4%, top tier 44.0% to 43.9%; error on the 10 wrongly routed listings 103% to 42%. The README examples do not move: the Uccle villa's luxury price sits under the cap.
Revisit when the luxury model is retrained on more rows or a lower threshold: the cap exists because it cannot price anything near €1M.

## 2026-11-24: no Hugging Face publishing

`hf/` (model card, Space image), `scripts/publish_hf.py` and its tests are gone, with `huggingface_hub` in the dev requirements. The demo is the Streamlit app, with the API on Render; the images are on Docker Hub and GHCR.
Why: Hugging Face now hosts Docker Spaces only for PRO accounts (the Space push got 402 Payment Required), and a second copy of the models and their card on the Hub was one more place for figures to go stale (the card was a month and two retrains behind).

## 2026-11-24: a "most likely" band inside the 80% range

The API returns `likely_range` next to `prediction_interval`, and the UI draws it darker inside the 80% range: the band that holds the price one time in two. It is the central 50% of the standard model's log residuals on the calibration split (`src/train.py::likely_band`), -12.1% / +14.6% around the estimate, kept inside the 80% range. Measured on the test split: 50.7% raw, 50.1% as served. `train()` writes it to `interval_calibration.json`; `python -m src.train --likely-band` added it to the committed file without retraining. The calibration split now serves the range margin and this band, still nothing that fits or tunes a model.
Why: the 80% range reads about -24% / +36% on average because it must cover the listings the model gets badly wrong, and on its own it looked like the model knew nothing. Narrowing it would have meant promising less than 8 in 10; the darker band shows the usual size without changing that promise.
