# Technical deep dive

How the price estimator works, why it is built this way, and what is still
weak. The README says what the service does and how to run it; this document
is for the person who has to change it.

Every figure below was measured by the scripts named next to it and is quoted
from the README or from `models/interval_calibration.json`. None was
re-measured for this document: re-running `src/train.py` is deterministic, but
it is the only way the figures can move, and it is a separate decision.

---

## 1. The request, end to end

```
Streamlit UI ──POST /predict──▶ FastAPI (api/app.py)
                                  │  pydantic validation (PropertyInput)
                                  ▼
                       api/predict.py :: PredictEngine.predict
                                  │  to_training_vocabulary()   src/features.py
                                  │  add_features()             src/features.py
                                  ▼
            ┌────────────┬────────────┬──────────────┬─────────────┐
            standard     luxury       luxury         10th / 90th
            regressor    regressor    classifier     quantile models
            └────────────┴────────────┴──────────────┴─────────────┘
                                  │
                       api/app.py :: routing (p ≥ 0.9 → luxury)
                                  │  range only for the standard segment
                                  ▼
                       JSON response  +  monitoring/logs.json
```

The five models are `sklearn.Pipeline`s that share one fitted preprocessor
(median imputation for numbers, most-frequent imputation then a
`TargetEncoder` for `property_type`, `province`, `city`, `property_state`).
All five run on every request; `api/app.py` then keeps one price.

| Artifact | Built by | Role |
|---|---|---|
| `models/pipeline.joblib` | `src/train.py` | Standard regressor, log-price, prices clipped to the 1st–99th percentile |
| `models/pipeline_lower.joblib`, `pipeline_upper.joblib` | `src/train.py` | 10th and 90th percentile models for the range |
| `models/luxury_classifier.joblib` | `src/train.py` | Probability that a property is above the 99th percentile (€1.9M) |
| `models/pipeline_luxury.joblib` | `src/train.py` | Regressor trained only on the luxury rows, unclipped prices |
| `models/luxury_threshold.json` | `src/train.py` | Luxury price (€1.9M) and routing threshold (0.9) |
| `models/interval_calibration.json` | `src/train.py` | Conformal margin for the range, with the coverage it reaches |
| `models/categories.json` | `scripts/export_categories.py` | The categories the models were fitted on |

---

## 2. The vocabulary contract

**The single most important rule in this repo: the model only understands the
exact strings it was trained on.** A `TargetEncoder` maps any string it never
saw to the global mean. Before the fix, the UI sent `HOUSE`, `Liège` and
`TO_RENOVATE` where training had `house`, `liege` and `TO RENOVATE`, so the
model silently ignored property type, province and condition (README, bug #1).

`src/features.py::to_training_vocabulary()` is the only gate between user input
and the encoder:

- property type, state and province are rewritten to the training spelling
  (any case, accents, underscores, French/Dutch/English province names); an
  unknown value raises `ValueError`, which the API turns into a 422;
- the city goes through `CITY_ALIASES` (80 French/Dutch/English pairs for the
  same municipality), then is looked up first in the user's province, then in
  the others;
- an unknown city is **kept** with a note rather than rejected, because the
  estimate still works on the province and the property; a city with fewer
  than `LOW_SAMPLE_CITY = 20` training rows gets a note too, because the
  encoder then leans on the global mean;
- it is idempotent: training-style input passes through unchanged, which is
  what lets the API and the training data share one function.

`models/categories.json` is read by both the API (to validate) and the UI (to
build its choice lists), so the UI can only offer values the model knows.

`add_features()` is the other shared function: imputed energy by state, total
surface defaulting to 1.2 × livable surface, `property_age`,
`property_state_encoded`. Train and inference call the same code, so they
cannot drift apart. The state dictionaries are keyed with spaces
(`"TO RENOVATE"`), the way the data spells them; see `docs/decisions.md`,
2026-10-10, for why that changed the numbers.

**Missing is not zero.** Optional fields default to *missing* and are imputed
as in training. They used to default to 0, a value that never occurs for plot
surface or distances, which moved a typical Liège house by about +14%
(README, bug #2). A plot surface of 0 is read as unknown for the same reason.

---

## 3. Training (`src/train.py`)

- **Data:** `data/clean/cleaned_data.json`, 15,746 listings scraped from
  Immovlan. It is git-ignored, so a fresh clone cannot retrain.
- **Split:** 80/20, `SPLIT_SEED = 42`, `TEST_SIZE = 0.2`. The evaluation scripts
  import these constants, so they score the same 3,150 validation rows that no
  model was fitted on. The standard model does use them for early stopping.
- **Target:** `log1p(price)`, clipped to the 1st–99th percentile for the
  standard and quantile models. The clip caps the standard model at €1.9M,
  which is why the luxury path exists.
- **Standard regressor:** XGBoost, `reg:absoluteerror`, hyperparameters from an
  Optuna search (`src/optimize.py`) hard-coded into `train.py`.
- **Luxury classifier:** XGBoost with `scale_pos_weight ≈ 99` (ratio of
  non-luxury to luxury rows). That weight inflates the probabilities, which is
  why the routing threshold sits as high as 0.9 (section 4).
- **Luxury regressor:** trained on the ~125 luxury rows of the training split,
  on **unclipped** prices. Training it on the clipped target made every luxury
  target €1.9M and the model a constant (README, bug #7); a test now fails if
  it becomes constant again.
- **Deterministic:** two runs give identical models, and the committed models
  are the ones `train.py` produces (README, bug #9).

---

## 4. Luxury routing

The classifier decides which regressor prices the property. On the validation
split (`scripts/evaluate_stratified.py`, README "Luxury Routing"):

| Routing | Overall MAPE | < €1M | €1M–€1.9M | ≥ €1.9M (35 rows) |
|---|---:|---:|---:|---:|
| None | 22.3% | 21.9% | 27.0% | 46.0% |
| **0.9 (chosen)** | **22.8%** | **22.3%** | **35.1%** | **34.9%** |

What a reader has to know:

- **Routing is kept for the top tier, not for the average.** It costs 0.5 point
  of overall MAPE and 8 points on the €1M–€1.9M tier, and it cuts the luxury
  bias from −46% to −27%.
- **0.9 is kept, not proven best.** With 35 luxury rows, one property moves
  the tier's MAPE by about 3 points; 0.85, 0.9 and 0.95 are within one or two
  properties of each other.
- **The switch is hard.** A property just under and just over the threshold can
  get prices €1M apart. The obvious fixes were measured and rejected
  (`scripts/evaluate_routing_blend.py`): a log-space blend weighted by the raw
  probability helps the €1.9M+ tier (34.9% → 23.8% MAPE) but pushes the
  < €1M bias from +6.6% to +19.6%, because that probability is inflated by
  `scale_pos_weight`; weighted by an isotonic-calibrated probability it helps
  the €1M–€1.9M tier (35.1% → 28.7%) and hurts the €1.9M+ tier (34.9% →
  36.4%). Neither beats the threshold where the threshold matters.

---

## 5. The price range

The standard segment returns a 10th–90th percentile range from two quantile
models. On unseen rows the raw models held the real price only ~71% of the
time, not 80% (README, bug #10).

`train.py` applies **conformalized quantile regression** (CQR):

1. split the standard-routed validation rows in two halves (`calibration_halves`);
2. on the first half, compute for each row how far the real log-price falls
   outside the raw range, and take the finite-sample quantile of those scores
   that gives 80% coverage (`conformal_margin`);
3. widen both bounds by that margin, and measure the coverage on the
   **second** half, which played no part in choosing it.

From `models/interval_calibration.json`:

| | Value |
|---|---:|
| Calibration rows / check rows | 1,554 / 1,556 |
| Log margin | +0.0445 (bounds × / ÷ 1.05) |
| Coverage on the check half, raw | 70.7% |
| Coverage on the check half, calibrated | 77.8% |

The README adds that the coverage averages 79.9% over 200 random halvings, so
77.8% is one draw below the target, not a systematic shortfall. No range is
returned for the luxury segment: the quantile models describe the standard
model's estimate, and ~125 rows are too few for a stable quantile fit.

---

## 6. Evaluation: what the numbers mean

All model figures are on the 3,150-row validation split. An earlier "holdout"
(rows absent from `data/training_baseline.csv`) still contained 3,938 of its
4,944 rows from the training split, which made MAPE look like 16.8% instead
of 22.4% and tuned the routing threshold on training data (README, bug #8).

Standard model alone (`monitoring/metrics.py`): MAE €92,632, MAPE 22.34%,
bias +4.62%. By tier, MAPE is 21.89% below €1M, 27.01% between €1M and €1.9M,
and 46.01% above €1.9M.

MAPE is the headline because a buyer thinks in percent of the price, but it
weights a €150k flat and a €3M villa equally. That is why every table is
split by price tier, and why the routing decision is read off the top tier's
bias rather than the overall MAPE.

---

## 7. Serving and deployment

- **API:** FastAPI, pydantic validation, `/` health check and `/ping` for
  UptimeRobot (it keeps Render's free tier from sleeping). Unknown categories
  give 422 with the list of accepted values.
- **UI:** Streamlit. Province and city sit outside `st.form` so the city list
  follows the province. `api_is_awake()` pings at most once a minute, and the
  predict timeout is 90 s because a sleeping Render instance takes up to a
  minute to wake.
- **Hosting:** API on Render, UI on Streamlit Community Cloud, both reading
  `$PORT`. Locally, `docker-compose.yml` runs both, the UI reaching the API by
  its service name (`API_URL=http://api:8000`).
- **Pinned libraries:** the models are pickles, so `scikit-learn`, `xgboost` and
  `category_encoders` are pinned to the versions they were pickled with. A
  newer `category_encoders` makes every prediction fail (README, bug #3). Any
  upgrade of those three means a retraining.
- **Failure paths:** the routing threshold is read from
  `models/luxury_threshold.json` with no fallback, so a missing file stops the
  API at start-up rather than routing with a guessed value. An unexpected
  error in `/predict` is logged with its traceback and answered with a generic
  500, never the exception text. `build_year` is bounded by today's date, not
  by a year written into the schema.
- **Tests:** 88, offline, run by `.github/workflows/tests.yml` on every push
  and pull request. They cover the vocabulary mapping, the real model's
  behaviour (house ≠ apartment, province and condition move the price, the
  luxury model is not constant), the conformal margin, the API contract and
  its failure paths, and the UI through Streamlit's `AppTest` with the API
  mocked.

---

## 8. Open issues, by point of view

These are found by reading the code. Each one is its own change.

### Software engineering

- **Prediction logs do not survive on Render** unless `PREDICTION_LOG_PATH`
  points at a persistent disk: Render's free-tier filesystem is wiped on each
  deploy and restart. The code side is done (JSON Lines, one appended line per
  prediction under a lock, with the engineered features); the hosting side is
  a configuration choice.
- **Import paths by `sys.path` edits** in `api/app.py`, `api/predict.py` and
  `streamlit/app.py`. They work, but they are why a stray
  `streamlit/__init__.py` once shadowed the Streamlit library (README, bug #5).
- **The training year is fixed, the calendar is not.** `CURRENT_YEAR = 2026`
  in `src/features.py` is the reference year the models learned
  `property_age` from, so it must stay put until a retraining. A property
  built after 2026 is accepted but gets a negative age, a value the models
  never saw.

### Data science

- **The thresholds are set on all rows.** The 1st/99th percentile clip of the
  target and `luxury_price_threshold` are taken on the full data, validation
  rows included. Measured: on the training split alone the 99th percentile is
  €1,890,000 instead of €1,900,000 and the 1st is €68,475 instead of
  €68,500. Fixing it changes the target, the luxury labels and every model, so
  it waits for the next retraining rather than moving the published figures.
- **Reproducible on one machine, not across machines.** Retraining twice in the
  same container gives identical models, and the old and new `train.py` give
  identical models there; but the committed standard and luxury models differ
  from that container's (up to 0.4 in log price), while the quantile models,
  the classifier and both JSON files match exactly. XGBoost's result depends
  on the CPU and thread count it ran on.
- **Two scripts, two sets of figures.** `monitoring/metrics.py` scores the
  standard model alone (below €1M: €67,657 MAE; the same model shows +6.3%
  bias in `evaluate_stratified.py`'s no-routing row);
  `scripts/evaluate_stratified.py` scores what the API serves, routing
  included (below €1M at 0.9: €70,377 MAE, +6.6% bias). The UI caption and the
  README's limitations quote the second; the README's performance table quotes
  the first. Both are right, as long as each figure says which one it is.
- **Luxury rows are few.** ~125 rows to fit a regressor and 35 to judge it:
  every luxury figure has a wide uncertainty that a single number hides.

### DRY

- The two Dockerfiles repeat the same install layers. Kept on purpose: Render
  builds `api/Dockerfile` by path, and a shared base image would add a build
  step to save six lines.

### GenAI

There is no LLM in this service, and the core estimate should stay a tabular
model: it is cheaper, deterministic and measurable. Where a single structured
LLM call could add value, with schema validation and a fallback:

- extracting surface, state or garage from a free-text listing description
  into `PropertyInput`;
- turning the response (segment, range, notes) into a plain-language
  explanation, without letting it invent a number.

### Strategy and business

- **What it is for:** a first estimate for a buyer or a seller, not a
  valuation. About 22% average error below €1M (≈ €68k) is useful to sort
  listings, not to set a price; the app says so next to every estimate.
- **Where it is weakest is where the money is.** Above €1.9M the error is
  around a third of the price, mostly underestimation.
- **The data ages.** One scrape, no refresh pipeline, and no real traffic has
  been logged durably yet (see above). A product version needs a
  scheduled re-scrape, retraining on a fixed cadence, and the same tier tables
  re-run each time.
- **Running cost is near zero** (free tiers), and the price of that is the
  cold start: up to a minute for the first estimate after a pause.
- **Scraped data:** reusing Immovlan listings commercially is a question for
  the site's terms, not for the code.
