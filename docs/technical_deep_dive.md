# Technical deep dive

How the price estimator works, why it is built this way, and what is still weak. The README says how to run it; this document is for the person who has to change it. Figures come from the scripts named next to them and from `models/interval_calibration.json`.

## 1. The request, end to end

```
Streamlit UI --POST /predict--> FastAPI (api/app.py)
                                  | pydantic validation (PropertyInput)
                                  v
                    api/predict.py :: PredictEngine.predict
                                  | to_training_vocabulary()   src/features.py
                                  | add_features(), add_location()
                                  v
        standard regressor | luxury regressor | luxury classifier | 10th/90th quantile models
                                  |
                    api/predict.py :: route (p >= 0.9 -> luxury)
                                  v
                    JSON response + prediction log (JSON Lines)
```

The five models are `sklearn.Pipeline`s sharing one preprocessor (median imputation for numbers, most-frequent imputation then a `TargetEncoder` for property type, province, city and state). All five run on every request; the API keeps one price.

| Artifact | Role |
|---|---|
| `pipeline.joblib` | Standard regressor, log price, prices clipped to the 1st-99th percentile |
| `pipeline_lower.joblib`, `pipeline_upper.joblib` | 10th and 90th percentile models for the range |
| `luxury_classifier.joblib` | Probability that a property is above the 99th percentile of training prices (about €1.95M) |
| `pipeline_luxury.joblib` | Regressor trained on the luxury rows only, unclipped prices |
| `luxury_threshold.json`, `interval_calibration.json` | Routing threshold (0.9), conformal margin of the range |
| `categories.json`, `locations.json` | Training vocabulary; median coordinates of each training postcode and city |

## 2. The vocabulary contract

**The model only understands the exact strings it was trained on.** A `TargetEncoder` maps any unseen string to the global mean. Once, the UI sent `HOUSE`, `Liège` and `TO_RENOVATE` where training had `house`, `liege` and `TO RENOVATE`, and the model silently ignored property type, province and condition.

`src/features.py::to_training_vocabulary()` is the only gate between input and encoder:

- type, state and province are rewritten to the training spelling (any case, accents, French/Dutch/English names); an unknown value raises `ValueError`, which the API answers with a 422;
- the city goes through `CITY_ALIASES` (80 French/Dutch/English pairs), then is looked up in the user's province first;
- an unknown city is kept with a note rather than rejected, and a city with fewer than 20 training rows gets a note too;
- it is idempotent, so the API and the training data share one function.

The rules for missing values, state keys, real columns and postcode locations, with their dates and the measured effects, are in [decisions.md](decisions.md).

## 3. Training (`python -m src.train`)

- **Data:** `data/clean/cleaned_data.json`, 15,746 listings scraped once from Immovlan. It is git-ignored, so a fresh clone cannot retrain. Asking prices under €10,000 are not prices and are dropped from every split by `MIN_PLAUSIBLE_PRICE`, a rule that reads the price alone, never a model's error.
- **Four disjoint splits** from `split_all()`, one seed: train 70% (fits everything), validation 10% (early stopping, Optuna, checking the routing threshold), calibration 5% (the range margin, nothing else), test 15% (sealed: only the final figures read it). Every script imports the same helpers.
- **Target:** `log1p(price)`, clipped to the training rows' 1st-99th percentile for the standard and quantile models, so the standard model cannot predict above about €1.95M. That is why the luxury path exists.
- **Standard regressor:** XGBoost, `reg:absoluteerror`, hyperparameters from a seeded Optuna search (`python -m src.optimize`, needs `pip install optuna`), hard-coded in `train.py`.
- **Luxury classifier:** XGBoost with `scale_pos_weight` near 98, which inflates its probabilities and explains the high threshold. **Luxury regressor:** the 111 luxury rows, on unclipped prices; training it on the clipped target once made it answer €1.9M for every property, and a test now fails if it becomes constant.
- **Reproducibility and the test split:** two runs on one machine give identical models, another machine can differ, so the committed models come from the `python:3.12` image with the pinned versions. Nothing is chosen by reading the test split. Both rules and their reasons are in [decisions.md](decisions.md).

## 4. Luxury routing

The classifier decides which regressor prices the property. On the test split (`python -m scripts.evaluate_stratified --test`):

| Routing | Overall MAPE | < €1M | €1M to €1.95M | >= €1.95M (21 rows) |
|---|---:|---:|---:|---:|
| None | 20.3% | 19.8% | 26.2% | 50.6% |
| 0.9, uncapped (until 2026-11-24) | 20.6% | 19.9% | 31.1% | 44.0% |
| 0.9, luxury price capped at 1.5x the standard one (deployed) | 20.3% | 19.8% | 27.4% | 43.9% |

- The luxury model only learned prices above €1.95M, so it never answers below about €2.2M. Uncapped, 10 of the 17 routed test listings were not luxury and got about twice their price (103% mean error on those 10). The routed price is now capped at 1.5 times the standard estimate (`src/train.py::routed_price`, chosen on the validation split): their error drops to 42%, the top tier keeps its gain (mean bias -50.6% without routing, -43.7% now), and routing no longer costs anything overall.
- 0.9 is kept, not proven best: it was picked on an earlier validation split and only checked on the next one. With 21 luxury test rows, one property moves the tier's MAPE by about 5 points.
- The switch is hard: two properties on either side of the threshold can differ by €1M. Probability blends were measured (`python -m scripts.evaluate_routing_blend`, validation split) and rejected: with the raw probability the top tier improves but the < €1M mean bias rises from +7.2% to +16.0%; with an isotonic-calibrated probability the gains do not hold on the half of the rows that played no part in the calibration.

## 5. The price range

The standard segment returns a 10th to 90th percentile range from two quantile models. Raw, they hold the real price 70.8% of the time on the test split, not 80%. `train.py` applies conformalized quantile regression: on the standard-routed calibration rows it takes the finite-sample quantile of how far the real log price falls outside the raw range, and widens both bounds by that margin (+0.0542 in log price, bounds multiplied and divided by 1.056). The guarantee needs calibration rows that played no part in fitting the bounds, so the calibration split serves nothing else.

Coverage on the 2,344 standard-routed test rows: 70.8% raw, 80.4% calibrated; over 200 random re-draws of the calibration and test rows, 80.2% on average (73.4% to 84.1%). The luxury segment gets no range: 111 rows are too few for a stable quantile fit.

## 6. Evaluation: what the numbers mean

- All headline figures are test-split figures. An earlier "holdout" held most of the training rows and made the error look 6 points better than it was.
- `python -m monitoring.metrics` scores the standard model alone (MAPE 20.28%, median 12.88%, MAE €85,137, R² 0.676 on euros, 0.81 on log price). `python -m scripts.evaluate_stratified --test` scores what the API serves, routing included (20.3%). Say which one a figure is.
- MAPE weights a €150k flat and a €3M villa equally, so every table is split by price tier.
- Mean and median bias answer different questions: an estimate at twice the price weighs +100%, one at half the price -50%, so the mean is pulled up by cheap listings. Below €1M, estimates run high under €150k (82% above the asking price, median +22%) and low between €500k and €1M (median -10%).
- 155 of the 2,361 test listings share an address with a training listing; on those the error is 10.7%, against 21.3% on the rest. The split is random by listing, not by address.
- Handling missing values differently (left to XGBoost, an "unknown" state) moved the error below €1M by 0.2 point at most, so it was left alone.

## 7. Serving and deployment

- **API:** FastAPI, `/ping` for UptimeRobot, which polls it so Render's free tier does not go to sleep ([screenshot](img/uptimeRobot.png)). The routing threshold is read from `models/luxury_threshold.json` with no fallback, so a missing file stops the service at start-up. A `/predict` failure is logged with its traceback and answered with a generic 500. `build_year` is bounded by today's date. `garden_m2` is deprecated and ignored: no model reads it, and it stays in the schema only so existing clients that send it still validate.
- **UI:** Streamlit; province and city sit outside `st.form` so the city list follows the province. The predict timeout is 120 s because a sleeping Render instance took 83 s to wake.
- **Hosting:** API on Render (a Python service, `uvicorn api.app:app --host 0.0.0.0 --port $PORT`), UI on Streamlit Community Cloud. Compose runs both, the UI reaching the API by service name.
- **Images:** the API image copies only `api/`, `src/`, `monitoring/` and `models/` and runs as a non-root user. The two Dockerfiles repeat their install layers on purpose: Render builds `api/Dockerfile` by path.
- **Pinned libraries:** the models are pickles, so scikit-learn, xgboost and category_encoders match the versions they were pickled with (category_encoders 2.10 or later breaks every prediction). Upgrading any of the three means retraining.
- **Imports:** modules are imported from the repository root (`api.app`, `src.features`) and run with `python -m`; pytest adds the root through `pyproject.toml`. Only `streamlit/app.py` edits `sys.path`, because `streamlit run` puts only `streamlit/` on it.

## 8. Known gaps

- `CURRENT_YEAR = 2026` in `src/features.py` is the year the models learned `property_age` from; it stays until a retraining. A property built after 2026 gets a negative age.
- The data ages: one scrape, no refresh pipeline, no durable production log. Reusing Immovlan listings commercially is a question for the site's terms.
- Luxury rows are few (111 to fit, 21 to judge), so every luxury figure has a wide uncertainty.
