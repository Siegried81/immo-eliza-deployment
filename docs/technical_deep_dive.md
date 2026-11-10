# Technical deep dive

How the price estimator works, why it is built this way, and what is still
weak. The README says what the service does and how to run it; this document
is for the person who has to change it.

Every figure below was measured by the scripts named next to it and is quoted
from the README or from `models/interval_calibration.json`. They come from the
models retrained on 2026-11-10 with a sealed test split (`docs/decisions.md`).

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
                       JSON response  +  monitoring/logs.jsonl
```

The five models are `sklearn.Pipeline`s that share one fitted preprocessor
(median imputation for numbers, most-frequent imputation then a
`TargetEncoder` for `property_type`, `province`, `city`, `property_state`).
All five run on every request; `api/app.py` then keeps one price.

| Artifact | Built by | Role |
|---|---|---|
| `models/pipeline.joblib` | `src/train.py` | Standard regressor, log-price, prices clipped to the 1st–99th percentile |
| `models/pipeline_lower.joblib`, `pipeline_upper.joblib` | `src/train.py` | 10th and 90th percentile models for the range |
| `models/luxury_classifier.joblib` | `src/train.py` | Probability that a property is above the 99th percentile of the training prices (€1.95M) |
| `models/pipeline_luxury.joblib` | `src/train.py` | Regressor trained only on the luxury rows, unclipped prices |
| `models/luxury_threshold.json` | `src/train.py` | Luxury price (€1.95M) and routing threshold (0.9) |
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
- **Split:** four disjoint parts from `split_all()`, `SPLIT_SEED = 42`. Train
  11,022 rows (fits everything), validation 1,575 (early stopping, the Optuna
  search, checking the routing threshold), calibration 787 (the range's margin
  only), test 2,362 (sealed: only the final figures read it). The test rows
  come from the old 80% training split, so the October threshold choice and
  search never scored them (a fresh 15% with the same seed would have sat
  inside the old 20% validation split). Every script imports these helpers,
  so they all see the same rows.
- **Target:** `log1p(price)`, clipped to the 1st to 99th percentile for the
  standard and quantile models, the percentiles taken on the training rows
  only (`price_bounds`). The clip caps the standard model at about €1.95M (€1,948,950),
  which is why the luxury path exists.
- **Missing build year:** stays missing in `add_features()`, like at the API,
  and the median imputer fills it with the training median (1974). It used to
  be 2000 at the API and the current year in training.
- **Standard regressor:** XGBoost, `reg:absoluteerror`, hyperparameters from a
  seeded Optuna search on the train and validation splits (`src/optimize.py`),
  hard-coded into `train.py`.
- **Luxury classifier:** XGBoost with `scale_pos_weight ≈ 98` (ratio of
  non-luxury to luxury training rows). That weight inflates the probabilities, which is
  why the routing threshold sits as high as 0.9 (section 4).
- **Luxury regressor:** trained on the 111 luxury rows of the training split,
  on **unclipped** prices. Training it on the clipped target made every luxury
  target €1.9M and the model a constant (README, bug #7); a test now fails if
  it becomes constant again.
- **Deterministic on one machine:** two runs give identical models, and the
  committed models are the ones `train.py` produces in the python:3.12 Linux
  image (README, bug #9; section 8).

---

## 4. Luxury routing

The classifier decides which regressor prices the property. On the sealed test
split (`scripts/evaluate_stratified.py --test`, README "Luxury Routing"):

| Routing | Overall MAPE | < €1M | €1M to €1.95M | ≥ €1.95M (21 rows) |
|---|---:|---:|---:|---:|
| None | 26.2% | 25.9% | 28.1% | 51.6% |
| **0.9 (deployed)** | **26.5%** | **26.0%** | **34.7%** | **46.2%** |

The first two columns carry one €3,048 listing, a data error worth 4 points
of MAPE on its own (section 6).

What a reader has to know:

- **Routing is kept for the top tier, not for the average.** It costs 0.3 point
  of overall MAPE and 7 points on the €1M to €1.95M tier, and it cuts the luxury
  bias from −51.6% to −43.0%.
- **0.9 is kept, not proven best.** It was picked on the old validation split
  and only checked on the new one, which has 19 luxury rows. There it cuts the
  luxury MAPE from 51.2% to 44.0%. With 21 luxury rows in the test split, one
  property moves the tier's MAPE by about 5 points.
- **The switch is hard.** A property just under and just over the threshold can
  get prices €1M apart. The obvious fixes were measured and rejected
  (`scripts/evaluate_routing_blend.py`, validation split): a log-space blend
  weighted by the raw probability helps the €1.95M+ tier (44.0% → 32.4% MAPE)
  but pushes the < €1M bias from +7.9% to +20.3%, because that probability is
  inflated by `scale_pos_weight`; weighted by an isotonic-calibrated
  probability it helps both upper tiers (30.7% → 24.6%, 44.0% → 39.5%) and
  costs the < €1M tier (23.0% → 23.6%). That's on 52 rows between €1M and
  €1.95M and 19 above, too few to switch.

---

## 5. The price range

The standard segment returns a 10th to 90th percentile range from two quantile
models. On the test split the raw models hold the real price 73.6% of the
time, not 80% (README, bug #10).

`train.py` applies **conformalized quantile regression** (CQR):

1. keep the calibration-split rows the API would give a range to (routed to
   the standard model);
2. compute for each row how far the real log-price falls outside the raw
   range, and take the finite-sample quantile of those scores that gives 80%
   coverage (`conformal_margin`);
3. widen both bounds by that margin, and measure the coverage on the
   standard-routed **test** rows.

The guarantee needs the calibration rows to be exchangeable with new ones, so
they must play no part in fitting the bounds. Until 2026-11-10 they were half
of the validation split, which also early-stopped the quantile models, so it
didn't strictly hold. The calibration split now serves nothing else.

From `models/interval_calibration.json`:

| | Value |
|---|---:|
| Calibration rows / test rows | 781 / 2,344 |
| Log margin | +0.0483 (bounds × / ÷ 1.049) |
| Coverage on the test split, raw | 73.6% |
| Coverage on the test split, calibrated | 80.9% |

Over 200 random re-draws of the calibration and test rows (same models), the
coverage averages 80.1% (76.2% to 84.7%). No range is returned for the luxury
segment: the quantile models describe the standard model's estimate, and 111
rows are too few for a stable quantile fit.

---

## 6. Evaluation: what the numbers mean

All model figures are on the 2,362-row test split, which nothing was fitted,
early-stopped, tuned or chosen on. The validation split did all of that, so
its scores are optimistic and never published as a headline (README, bug #13).
An earlier "holdout" (rows absent from `data/training_baseline.csv`) was worse:
it still contained 3,938 of its 4,944 rows from the training split, which made
MAPE look like 16.8% instead of 22.4% on the old validation split (README,
bug #8).

Standard model alone (`monitoring/metrics.py`): MAE €92,024, MAPE 26.24%,
median APE 14.85%, R² 0.646 (euros), bias +7.88%. By tier, MAPE is 25.94%
below €1M, 28.13% between €1M and €1.95M, and 51.59% above €1.95M.

One test row is a data error: a 77 m² Oostende apartment at €3,048, priced
€297,380 by the model. That's a 9,657% error, and on its own it adds 4 points
to the MAPE (22.17% without it) and 4 points to the bias (+3.79% without it).
It stays in: removing a row after seeing the score is a choice made on the
test split. The cleaning step should drop prices like that, and the next
retraining will need a new test split once it does.

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
  its service name (`API_URL=http://api:8000`). The Hugging Face Space
  (`hf/space/`) runs both in one container instead, because a Space exposes a
  single port: the API listens on localhost, the UI on 7860, and `start.sh`
  waits for `/ping` before starting the UI so its cached health check does not
  record the API as asleep while the models load. `API_DOCS_URL=""` hides the
  docs link there, since the API is not reachable from outside. That Space is
  built and run locally with Docker but not published: Hugging Face hosts
  Docker Spaces for PRO accounts only.
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
- **Tests:** 106, offline, run by `.github/workflows/tests.yml` on every push
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

- **Reproducible on one machine, not across machines.** Retraining twice in the
  same container gives identical models. The models trained on the Windows
  workstation differed from the container's (standard and luxury regressors,
  up to 0.4 in log price), because XGBoost's result depends on the CPU and
  thread count it ran on. The committed models are now the ones trained in the
  python:3.12 Linux image, the image the service is deployed with, so
  `python src/train.py` run in that image reproduces them. The 2026-11-10
  models were trained in python:3.12 with the pinned versions.
- **Two scripts, two sets of figures.** `monitoring/metrics.py` scores the
  standard model alone (26.24% MAPE overall); `scripts/evaluate_stratified.py
  --test` scores what the API serves, routing included (26.5% overall). Below
  €1M they nearly agree on the test split (€68,579 and €69,601 MAE), because
  0.9 routes one of those 2,254 rows. The UI caption and the
  README's limitations quote the second; the README's performance table quotes
  the first. Both are right, as long as each figure says which one it is.
- **Luxury rows are few.** 111 rows to fit a regressor and 21 to judge it:
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
  valuation. About 22% average error below €1M (≈ €70k as deployed, without the one bad test row) is useful to sort
  listings, not to set a price; the app says so next to every estimate.
- **Where it is weakest is where the money is.** Above €1.95M the error is
  close to half the price (46.2% on the test split), mostly underestimation.
- **The data ages.** One scrape, no refresh pipeline, and no real traffic has
  been logged durably yet (see above). A product version needs a
  scheduled re-scrape, retraining on a fixed cadence, and the same tier tables
  re-run each time.
- **Running cost is near zero** (free tiers), and the price of that is the
  cold start: up to a minute for the first estimate after a pause.
- **Scraped data:** reusing Immovlan listings commercially is a question for
  the site's terms, not for the code.
