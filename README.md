# Immo Eliza Deployment

Deployment phase of the **Immo Eliza Machine Learning** project: a REST API + web interface that predicts Belgian residential property prices, with a dedicated luxury-segment model for high-end properties.

- 🚀 FastAPI backend + 🎨 Streamlit frontend
- 🤖 XGBoost regression, with a luxury-segment model + routing classifier
- ✅ Automated pytest suite (129 tests, API and UI, no network, run in CI)
- 📊 Prediction logging, PSI-based drift detection
- 📏 Headline figures on a sealed test split (2,361 listings) that nothing was fitted, tuned or chosen on
- 🔎 Audited end to end: [19 bugs found and fixed](#bugs-found-and-fixed), including one that made the model ignore property type and province and a luxury model that always answered €1.9M
- 📘 [Technical deep dive](docs/technical_deep_dive.md): how each model, the routing and the price range work, why, and what is still weak

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Bugs Found and Fixed](#bugs-found-and-fixed)
3. [File Purpose](#file-purpose)
4. [Running the Application](#running-the-application)
5. [Reliability & Uptime Monitoring](#reliability--uptime-monitoring)
6. [Using the Application](#using-the-application)
7. [Prediction Examples](#prediction-examples)
8. [Model Performance & Monitoring](#model-performance--monitoring)
9. [Luxury Routing](#luxury-routing)
10. [Known Limitations & Disclaimer](#known-limitations--disclaimer)
11. [Drift Analysis](#drift-analysis)
12. [Automated Testing](#automated-testing)
13. [Future Improvements](#future-improvements)
14. [Technologies Used](#technologies-used)
15. [Conclusion](#conclusion)

---

## Project Architecture

```mermaid
flowchart LR
    U[User] --> S[Streamlit UI<br/>Streamlit Cloud]
    S -- POST /predict --> A[FastAPI<br/>Render]
    A --> N[to_training_vocabulary<br/>src/features.py]
    N --> F[add_features]
    F --> C{Luxury classifier<br/>p ≥ 0.9?}
    C -- no --> M1[Standard XGBoost<br/>+ 10th/90th quantile models]
    C -- yes --> M2[Luxury XGBoost]
    A --> L[(monitoring/logs.jsonl)]
    L --> D[PSI drift report]
    V[(models/categories.json)] --> N
    V --> S
```

```text
immo-eliza-deployment/
├── .github/workflows/
│   ├── tests.yml            # runs the test suite on every push and PR
│   └── publish.yml          # builds and publishes the API and UI images
├── api/
│   ├── __init__.py
│   ├── app.py               # FastAPI app, request validation, luxury routing
│   ├── Dockerfile
│   └── predict.py            # Loads model artifacts, runs inference
├── data/
├── docs/
│   ├── decisions.md           # dated record of changes to what a number means
│   └── technical_deep_dive.md # how it works, why, open issues
├── hf/
│   ├── model_card.md          # README of the Hugging Face model repo
│   └── space/                 # Dockerfile, start.sh and README of the Hugging Face Space
├── models/                    # pickled pipelines + categories.json (training vocabulary) + locations.json (centroids)
├── monitoring/
│   ├── __init__.py
│   ├── check_drift.py         # PSI drift report: logged features vs. training split
│   ├── generate_logs.py       # Synthetic log (validation rows) to test the drift report
│   ├── metrics.py             # Standard model on the sealed test split, by price tier
│   └── monitor.py             # JSON Lines prediction log, PSI computation
├── scripts/
│   ├── evaluate_stratified.py  # Threshold sweep (validation) or final figures (test), per tier
│   ├── evaluate_routing_blend.py # Threshold vs probability blends, per tier (read-only)
│   ├── export_categories.py    # Writes models/categories.json after each retraining
│   ├── predict_cli.py          # Terminal predictions, same engine and routing as the API
│   └── publish_hf.py           # Publishes the Space or the model repo to Hugging Face
├── src/
│   ├── __init__.py
│   ├── features.py            # Single source of truth for feature engineering
│   ├── optimize.py            # Optuna hyperparameter search
│   └── train.py               # Full training workflow (standard + luxury + routing + quantile models)
├── streamlit/
│   ├── app.py
│   └── Dockerfile
├── tests/
│   ├── conftest.py            # disables prediction logging during tests
│   ├── test_app.py
│   ├── test_monitoring_and_tools.py # prediction log, drift report, CLI, shared split
│   ├── test_predict.py        # vocabulary mapping + real-model behaviour
│   ├── test_publish_hf.py     # what each Hugging Face repo receives, Hub mocked
│   ├── test_robustness.py     # failure paths: config, 500s, build year, quantile config
│   ├── test_routing_blend.py  # the blend measured by evaluate_routing_blend.py
│   └── test_streamlit_app.py  # UI through Streamlit's AppTest, API mocked
├── .streamlit/config.toml     # UI theme
├── .dockerignore
├── .env.example               # every environment variable, with its default
├── .gitattributes             # keeps shell scripts LF for the Linux containers
├── .gitignore
├── docker-compose.yml
├── docker-compose.debug.yml   # same stack under debugpy, for VS Code
├── README.md
├── requirements.txt
└── requirements-dev.txt       # debugpy, huggingface_hub; never in the published images
```

> `scripts/evaluate_stratified.py` reports metrics **per price tier** and sweeps the luxury-routing threshold. It replaces the old global-metric `evaluate.py`.

---

## File Purpose

| File | Purpose |
|---|---|
| `api/app.py` | FastAPI endpoints, request validation, luxury routing decision (via `luxury_threshold.json`), `/ping` keep-alive for UptimeRobot |
| `api/predict.py` | Loads the 5 model artifacts (standard, luxury, luxury classifier, lower/upper quantile) and runs inference |
| `monitoring/monitor.py` | Appends each prediction, with the features the models read, to a JSON Lines log (`PREDICTION_LOG_PATH` to move it); computes PSI |
| `monitoring/check_drift.py` | PSI of the logged features against `train.py`'s training split; reports how many entries are real traffic |
| `monitoring/metrics.py` | Scores `pipeline.joblib` on `train.py`'s sealed **test split**, overall and by price tier |
| `monitoring/generate_logs.py` | Writes validation rows to `monitoring/synthetic_logs.jsonl` (never the real log) to test the drift report |
| `src/features.py` | Single source of truth for feature engineering, imported by every script to prevent train/inference skew. `to_training_vocabulary()` rewrites user input into the exact category strings the models were fitted on |
| `models/interval_calibration.json` | Conformal margin applied to the price range (set on the calibration split), with the coverage before and after on the test split |
| `models/categories.json` | Provinces, cities (with training row counts), property types and states the models know; read by the API and the UI |
| `scripts/export_categories.py` | Regenerates `models/categories.json` from the training split |
| `models/locations.json` | Median coordinates of each training postcode and city, and each city's usual postcode; written by `src/train.py`, read by the API (to place a request) and the UI (to start the postcode on the city's) |
| `src/train.py` | Full training workflow: cleans data, caps outliers, trains standard model, luxury model, routing classifier, and quantile (interval) models; writes `luxury_threshold.json`. Deterministic on a given machine: two runs give identical models (another machine can differ, see `docs/decisions.md`, 2026-10-12). `load_dataset()`, `split_all()` (train, validation, calibration, test), `split()`, `clipped_log_target()` and `price_tiers()` are shared with every script |
| `src/optimize.py` | Seeded Optuna hyperparameter search on the train and validation splits, scored on exactly the setup `train.py` ships |
| `scripts/evaluate_stratified.py` | Full routing pipeline per price tier: a threshold sweep on the validation split (where a threshold gets checked), or with `--test` the deployed threshold on the sealed test split (the figures to publish) |
| `scripts/evaluate_routing_blend.py` | Read-only comparison of the 0.9 threshold with probability blends of both models, per tier |
| `scripts/predict_cli.py` | Terminal predictions through the API's schema, engine and routing |
| `scripts/publish_hf.py` | Publishes the app as a Hugging Face Space or the models as a Hub model repo; only git-tracked files plus `hf/`, dry run unless `--push` |
| `hf/space/` | The Space's `Dockerfile`, `start.sh` (API on localhost, then the UI on port 7860) and `README.md` (Space settings) |
| `hf/model_card.md` | The model repo's card: files, usage, evaluation and limitations |
| `streamlit/app.py` | Web UI: choice lists from `categories.json`, estimate with its likely range, price per m², segment badge, caveats and a history of recent estimates |
| `tests/` | API, vocabulary mapping, real-model behaviour and UI tests (see [Automated Testing](#automated-testing)) |
| `docker-compose.yml` | Orchestrates API + Streamlit containers |
| `docs/technical_deep_dive.md` | How it works and why: vocabulary contract, training, routing, conformal range, evaluation, open issues by point of view |
| `docs/decisions.md` | Dated record of every change that altered what a number means |

> **Legacy note:** `pipeline.joblib` now bundles preprocessor + model in one `sklearn.Pipeline`, replacing the old separate `best_XGBoost.json` / `preprocessor.joblib` files.

---

## Bugs Found and Fixed

An end-to-end audit of the deployed service (not just the notebook metrics) found these issues:

| # | Problem | Impact | Fix |
|---|---|---|---|
| 1 | The UI sent `HOUSE`, `Liège`, `TO_RENOVATE`; the TargetEncoder was fitted on `house`, `liege`, `TO RENOVATE`. Unseen strings fall back to the global mean. | **The model ignored property type, province and condition.** A house and an apartment with the same inputs got the exact same price, and "to renovate" was priced above "normal". | `to_training_vocabulary()` maps every spelling (English/French/Dutch, any case, accents) to the training value; unknown categories return 422. Tests pin "house ≠ apartment", "to renovate < normal", "Walloon Brabant > Liège". |
| 2 | Optional fields defaulted to `0`, a value that never appears in training for plot surface or distances (they were missing and imputed instead). | Out-of-distribution input: about **+14%** on a typical Liège house. | Optional fields default to *missing* and are imputed exactly as in training; a plot surface of 0 is read as unknown. |
| 3 | `requirements.txt` did not pin the libraries the models were pickled with. | A fresh install (Render, Streamlit Cloud) gets `category_encoders` ≥ 2.10 and every prediction fails with `AttributeError`. | Pinned `scikit-learn==1.9.0`, `xgboost==3.2.0`, `category_encoders==2.8.1` (versions read from the pickles). |
| 4 | `docker-compose.yml` was UTF-16, mapped `8000:1000`, mounted `./api` over `/app` (hiding `src/`, `monitoring/`, `models/`) and never told the UI where the API was. | `docker compose up` could not work. | Rewritten: correct ports, no source volumes, `API_URL=http://api:8000`, restart policy. Dockerfiles read Render's `$PORT`. |
| 5 | `streamlit/app.py` had a broken indentation (`try:` at column 0), and an empty `streamlit/__init__.py` shadowed the `streamlit` library whenever the repo root was on `sys.path`. | The UI crashed on start. | New UI; the stray package file is removed. |
| 6 | The price range comes from the standard-segment quantile models but was returned next to luxury-model prices. | A range describing a different estimate. | The range is only returned for the standard segment. |
| 7 | `src/train.py` clipped prices at the 99th percentile, then trained the luxury regressor on rows *above* that same percentile. | Every luxury target was €1.9M: **the luxury model answered €1.9M for every property**, so a 400 m² Brussels house jumped from ~€640k (300 m²) to €1.9M. | The luxury regressor is trained on unclipped prices. A test fails if it ever becomes constant again. |
| 8 | The "true holdout" in `metrics.py` and `evaluate_stratified.py` (rows absent from `training_baseline.csv`) still contained 3,938 of its 4,944 rows from the training split, including 23 of the 29 €3M+ properties. | Metrics were optimistic (16.8% MAPE instead of 22.4%) and rewarded memorisation, so the routing threshold was tuned on training data. | Both scripts score `train.py`'s validation split (3,150 rows no model was fitted on); the threshold was re-chosen there (0.7 → 0.9). |
| 9 | The committed `pipeline.joblib` could not be reproduced by `src/train.py`. | No way to know how the deployed model was built. | All five artifacts are retrained by `src/train.py` (same validation MAPE: 22.42% → 22.39%) and two runs on the same machine give identical models. |
| 10 | The "80%" price range (10th to 90th percentile models) held the real price only ~71% of the time on unseen listings. | Users were told the range was more reliable than it is. | Conformalized quantile regression: `train.py` widens both bounds by a margin computed on the calibration split and measures the result on the test split (70.8% → 80.4%, `models/interval_calibration.json`; 80.2% on average over 200 random re-draws). Until 2026-11-10 it used two halves of the validation split, see #13. |
| 11 | The API defaulted a left-out `bedroom_count` to 1, while a training row without one was imputed with the median (3). | An API or CLI request without a bedroom count was priced as a one-bedroom: **-15.6%** on a 150 m² Liège house (€251,749 → €212,429), -7.0% on an 85 m² Elsene apartment. The UI always sends the field, so it was not affected. | `bedroom_count` defaults to *missing*, like the other optional fields, and goes to the pipeline's median imputer. |
| 12 | `add_features()` filled a column the caller did not give with 0 (the three distances) or 1 (bedrooms). Training rows always have those columns, so only a direct `engine.predict()` call reached these defaults, such as the example in the Hugging Face model card. | The same 90 m² Ixelles apartment (built in 2000) got **€300,692 from the engine and €363,063 from the API** (-17.2%): the engine read distances of 0 m and one bedroom. | The defaults are *missing*, so the median imputer fills them as in training and the engine agrees with the API. |

| 13 | Every published figure (22.5% MAPE, 23.0% as deployed, 77.9% range coverage) came from the 3,150-row validation split. Those rows also early-stopped the standard and quantile models, scored the Optuna search and picked the 0.9 routing threshold. The range was calibrated on half of them, rows the quantile models had early-stopped on. | The figures were validation scores, so optimistic, and the range's coverage guarantee didn't strictly hold. | Four disjoint splits (`split_all()` in `src/train.py`): train 70%, validation 10% (early stopping, Optuna, threshold), calibration 5% (the range margin only), test 15% (sealed, final figures only). The test rows come from the old 80%, so the old threshold choice never saw them. Optuna re-run, all five models retrained, every figure re-measured on the test split. |
| 14 | A left-out build year had two rules: the API sent 2000, while `add_features()` (and so training) used the current year, a new build. | The same 150 m² Liège house was €251,749 or €209,990 depending on the path. | One rule: a missing year stays missing and the pipeline's median imputer fills it with the training median (1974). Retrained with it. |
| 15 | The cleaned data names the energy column `energy_consumption_kWh/m2/year`; the code read `energy_consumption_kWh_m2_year`. `add_features()` found no such column, created it at 0 and filled every row with the figure for its state. | **The models never read a real energy consumption**, though 12,841 of 15,746 listings give one. To them the field was a second code for the state (80 = new, 300 = to demolish), so a user who typed 250 kWh, close to the median listing (264), was priced like a ruin: **-16%** on a 150 m² Liège house (€220,580 → €184,617). | `to_model_columns()` renames the column and refuses listings with a model input missing. A value of 0 or above 1,000 kWh counts as not given. The figure used when none is given is now the median of the training listings in that state (260 kWh for a normal state, was 130). Optuna re-run, all five models retrained: below €1M the test error went from 21.8% to 21.3% (without the bad row) with this fix alone, and 250 kWh now prices within 5% of a consumption left out. |
| 16 | `postcode` is required by the API and the form, and no model read it. The city was the only location, through one target-encoded number. | Two houses in the same city and different postcodes got the same price, and the models knew nothing of which cities are neighbours. | The models read the postcode and the median coordinates of its training listings (`add_location()`, `models/locations.json`; the city's when the postcode is new). Trained on centroids, not each listing's own coordinates, because a request has no coordinates. In cross-validation this took a point off the error below €1M (21.7% → 20.7%). The form starts the postcode on the city's usual one instead of 1000. |
| 17 | Three listings are priced under €10,000 (a 209 m² house at €2,500, another at €5,000, a flat at €3,048), then nothing until €19,900. The flat sat in the test split, where its 9,430% error added 4 points to the average error. | Two training rows taught the models a price that isn't one, and every headline figure needed a "without one bad row" footnote. | `MIN_PLAUSIBLE_PRICE = 10_000` in `src/train.py`: `split_all()` drops those rows from every split, after the draw, so no other row changes split. The rule reads the price alone, never a model's error. |
| 18 | The API defaulted a left-out `terrace` to 0. 5.7% of the training listings don't say whether there is one, and the median imputer filled them with 1. | Replaying 364 test listings through the API, the 24 with an unknown terrace came out up to **13% lower** than the pipeline that was scored (4.5% at the median); the 340 others matched exactly. On a 150 m² Liège house: €236,847 left out, €241,607 with a terrace. The UI always sends the field, so it was not affected. | `terrace` defaults to *missing*, like the other optional fields; send 0 for "no terrace". `add_features()` no longer fills an absent terrace or surface with 0. The replay now matches on all 364. |
| 19 | `models/locations.json` filed cities by name alone, and nine names exist in two provinces (Halle, Deurne, Vorst, Perwez, Celles...). Introduced with bug #16's fix, found by the audit the same day. | The form started Antwerp's Vorst on postcode 1190 (Brussels) and Flemish Brabant's Deurne on 2100 (Antwerp), so the estimate was for the other town unless the user corrected it. | Cities are filed under their province. No model changed: training rows always find their own postcode, so a full retraining gives the same five models, bit for bit. |

Tests also wrote every request into `monitoring/logs.json`, polluting the drift report; logging is now disabled in tests.

---

## Running the Application

### Live Deployment

- **Web App (Streamlit Community Cloud):** https://immo-eliza-deployment-sieg.streamlit.app
- **API (Render):** https://immo-eliza-deployment-j8av.onrender.com (docs at `/docs`).
  It runs as a Render Python service, not from the Dockerfile:
  `uvicorn api.app:app --host 0.0.0.0 --port $PORT`. On the free tier it sleeps when
  idle, and the first request after that took 83 s on 2026-11-06.
- **Models (Hugging Face Hub):** https://huggingface.co/Siegried81/immo-eliza-price-models

To republish the models after retraining (needs `hf auth login` with a write token):

```bash
python scripts/publish_hf.py model           # dry run: lists what would be uploaded
python scripts/publish_hf.py model --push    # the model artifacts and their card
```

`hf/space/` holds a Hugging Face Space that runs the API and the UI in one
container (`python scripts/publish_hf.py space --push`). It is built and tested
locally but not published: Hugging Face only hosts Docker Spaces for PRO
accounts (the Hub answered 402 to the creation request on 2026-10-25).

### Method 1: Docker Compose (recommended)

```bash
docker compose up --build
```

App available at `http://localhost:8508`, API at `http://localhost:8003/docs`.

### Method 2: Pre-built images from Docker Hub

No build and no clone needed. The images are `siegried/immo-eliza-api` and
`siegried/immo-eliza-ui`, run as containers `api` and `ui` on a network named
`immo`. The exact commands are in [DOCKERHUB.md](DOCKERHUB.md#how-to-use-this-image).

Methods 1 and 2 publish the same ports, 8003 and 8508, so only one stack can run at a time.
Stop the other first: `docker compose down` before Method 2, or
`docker rm -f api ui` before Method 1.

### Method 3: Manual (development)

Python 3.12 recommended (the pinned scikit-learn needs ≥ 3.11).

```bash
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8003                                 # terminal 1
API_URL=http://localhost:8003 streamlit run streamlit/app.py --server.port 8508 # terminal 2
```

### Method 4: Retrain & re-evaluate

```bash
python src/train.py                     # trains all 5 artifacts (deterministic on one machine)
python scripts/export_categories.py     # refreshes models/categories.json
python scripts/evaluate_stratified.py   # sweeps the luxury-routing threshold on the validation split
python scripts/evaluate_stratified.py --test   # deployed pipeline on the sealed test split
python monitoring/metrics.py            # standard-model metrics by tier on the test split
```

---

## Reliability & Uptime Monitoring

- **Docker** auto-restarts either container on crash.
- **`/ping`** endpoint: lightweight liveness check.
- **UptimeRobot** polls `/ping` to prevent cold starts on free-tier hosting (see `tests/uptimeRobot.png`).

---

## Using the Application

1. Pick the province, then the city: only cities present in the training data are offered, and the UI warns when a city has fewer than 20 listings (the target encoder then leans on the province).
2. Describe the property. Garage and terrace are yes/no, as in the training data. Leave the plot surface at 0 if unknown, and the energy consumption off if unknown: both are then estimated as during training.
3. Click **Estimate**. You get the price, its likely range, the price per m², the luxury probability and the caveats that apply. The last five estimates stay visible for comparison.

The sidebar shows whether the API is awake: on Render's free tier the first call after a pause can take up to a minute.

---

## Prediction Examples

> These screenshots show the **previous** UI, before bug #1 was fixed: the prices below were computed while the model ignored property type, province and condition, and will differ with the current version.

**Example 1: standard property**

![Prediction 1](tests/predict1.png)

Estimated price: **€283,578**

**Example 2: higher-value property**

![Prediction 2](tests/predict2.png)

Estimated price: **€1,449,607**

---

## Model Performance & Monitoring

The data (15,746 listings, 15,743 once the three priced under €10,000 are dropped) is cut into four splits with one seed (`split_all()` in `src/train.py`):

| Split | Rows | Used for |
|---|---:|---|
| Train | 11,020 | Fitting every model and the preprocessor |
| Validation | 1,575 | Early stopping, the Optuna search, checking the routing threshold |
| Calibration | 787 | The price range's conformal margin, nothing else |
| Test | 2,361 | Nothing but the figures below |

All figures below are on the **test split**. Nothing was fitted, early-stopped, tuned or chosen on it. Until 2026-11-10 this README quoted the validation split, which did all of that (bug #13, `docs/decisions.md`).

**Standard model alone** (`monitoring/metrics.py`):

| Metric | Value |
|---|---:|
| MAE | €85,137 |
| RMSE | €201,846 |
| MAPE | 20.28% |
| Median APE | 12.88% |
| R² (euros) | 0.676 |
| Mean bias | +3.44% |
| Median bias | −0.37% |

| Tier | N | MAE | MAPE |
|---|---:|---:|---:|
| < €1M | 2,253 | €62,512 | 19.77% |
| €1M to €1.95M | 87 | €362,890 | 26.20% |
| €1.95M+ | 21 | €1,361,801 | 50.60% |

**The €3,048 listing is gone, by a rule.** Until 2026-11-15 the test split held a 77 m² apartment in Oostende listed at €3,048. One row, a 9,430% error, 4 points of MAPE. Removing it because it hurt the score would have been choosing on the test, so it stayed. It now leaves under a rule that reads the price alone and applies to every split: an asking price under €10,000 is not the price of a home (bug #17). The rule also takes two training rows, and it was prompted by a row seen in the test split, which `docs/decisions.md` says plainly.

**Two biases, and they disagree.** The mean bias says +3.4%, the median says −0.37%: half the estimates are above the asking price, half below. The mean is not a typical estimate. It averages (estimate − price) / price, where an estimate at twice the price weighs +100% and one at half the price −50%, and it is carried by the cheapest listings. As deployed, below €1M (`scripts/evaluate_stratified.py --test` prints the three readings of the bias per tier):

| Asking price | Listings | Average error | Median error | Median bias | Estimates above the price |
|---|---:|---:|---:|---:|---:|
| Under €150k | 257 | 46.2% | 22.8% | +22.4% | 82% |
| €150k to €250k | 673 | 17.0% | 11.7% | +4.3% | 59% |
| €250k to €500k | 857 | 14.0% | 10.4% | −1.4% | 45% |
| €500k to €1M | 466 | 20.8% | 17.1% | −9.8% | 31% |

The model pulls every price towards the middle of the market. One listing in nine is under €150k, often a ruin, a share or a listing with no surface and no state given. Handling missing values differently (left to XGBoost, an "unknown" state, a count of missing fields) was measured in cross-validation and moved the error below €1M by 0.2 point at most, so it was left alone: nothing in the inputs tells a ruin from a house.

The standard model is trained on prices clipped to the 1st to 99th percentile of the training rows (€69,000 to €1,949,050), so it can't predict above about €1.95M. That's what the luxury model is for.

**Prediction interval:** standard-segment predictions come with a 10th to 90th percentile range from two extra `XGBRegressor` models (`objective="reg:quantileerror"`), returned as `prediction_interval: {lower, upper}`. The raw models are too narrow: on the test split they hold the real price 70.8% of the time. `train.py` calibrates them with **conformalized quantile regression** (CQR): on the calibration split (standard-routed rows, real unclipped prices) it finds the log-scale margin that makes 80% of prices fall inside, saves it to `models/interval_calibration.json`, and the API widens both bounds by it (currently ×1.056 / ÷1.056). The calibration rows never served early stopping, so the coverage guarantee holds. On the test split, coverage goes from 70.8% to 80.4% (2,344 rows). Not available for the luxury segment (too few rows for a stable quantile fit).

---

## Luxury Routing

A routing classifier estimates whether a property is in the top 1% of prices (above €1,949,050, the 99th percentile of the training rows); above the routing threshold, a dedicated luxury regressor trained on those 111 listings prices it instead of the standard model.

Sealed test split, by tier (`scripts/evaluate_stratified.py --test`):

| Routing | Overall MAPE | < €1M | €1M to €1.95M | ≥ €1.95M (21) | Rows routed |
|---|---:|---:|---:|---:|---:|
| No routing | 20.3% | 19.8% | 26.2% | 50.6% (bias −50.6%) | 0 |
| **Threshold 0.9 (deployed)** | **20.6%** | **19.9%** | **31.1%** | **44.0%** (bias −41.3%) | **17** |

**Why 0.9:** the classifier is trained with `scale_pos_weight ≈ 98`, which inflates its probabilities, so the threshold has to be high. 0.9 was picked in October on the old validation split and checked, not re-tuned, on the new one (19 listings above €1.95M, too few to tune on). There, 0.9 cuts the top tier's error from 50.5% to 40.8% (bias −50.5% to −34.7%), costs 0.2 point below €1M and 4 points between €1M and €1.95M. On the test split it does the same: the top tier goes from 50.6% to 44.0%, the €1M to €1.95M tier from 26.2% to 31.1%, and overall MAPE rises 0.3 point. It's kept for the top tier, not for the average. With 21 test listings in that tier, one property moves its MAPE by about 5 points.

---

## Known Limitations & Disclaimer

This is an **estimation tool, not a valuation system**.

- **Accuracy:** as deployed (routing at 0.9), on the 2,361 test listings: 20.6% average error, 12.9% median. Below €1M it's 19.9% (≈ €64k) and 12.5% median: half the estimates land within 13% of the asking price, two in three within 20%.
- **Cheap and expensive listings:** estimates run high under €150k (82% of them above the asking price, median +22%) and low between €500k and €1M (69% below, median −10%). Between €250k and €500k they are balanced. The mean bias below €1M (+5.1%) comes from the cheap end; the median bias is +0.3%.
- **Same building, both sides of the split:** 155 of the 2,361 test listings share their address with a training listing (flats in one development, one address appears 33 times in the data). On those the error is 10.7%; on the 2,206 others it's 21.3% (20.6% below €1M, 13.4% median). For a property in a building the models have never seen, read the second figure. The split is random by listing, not by address.
- **Location is the postcode's centroid**, not the street: two houses in one postcode get the same location. A postcode with no training listing (about 1% of the test listings) falls back to the city's centroid, and the response says so. Figures from `scripts/evaluate_stratified.py --test`; the standard model alone is in [Model Performance](#model-performance--monitoring).
- **High-end properties (€1.95M+):** the luxury model learned from 111 listings and is judged on 21; expect errors around half the price, typically underestimation. The standard model underprices expensive homes systematically (−50.6% on this tier without routing); routing cuts that bias to −41.3%, it doesn't remove it.
- **Routing trade-off:** routing helps the luxury tier but costs accuracy between €1M and €1.95M (see [Luxury Routing](#luxury-routing)).
- **Range:** calibrated to hold the real price 8 times in 10 (80.4% on the test split; 80.2% on average over 200 random re-draws of the calibration and test rows). For a given property it can be wide (median width ≈ €192k).
- **Build year left out:** priced at the training median, 1974. On a 150 m² Liège house (postcode 4000) that's €236,745, against €284,214 if you say 2000. Send the year when you know it.
- **City names:** the training data keeps one spelling per municipality (e.g. *Elsene*, not *Ixelles*). `CITY_ALIASES` in `src/features.py` maps 80 French, Dutch and English names to that spelling; a name outside both lists is accepted but ignored, and the API says so in `notes`.

These limitations are documented here and surfaced in the Streamlit app as a disclaimer next to each prediction, plus a flag when routed to the luxury model.

---

## Drift Analysis

Every prediction the API serves is appended to a JSON Lines log (`monitoring/logs.jsonl`, or `PREDICTION_LOG_PATH`) with the engineered features the models read. `monitoring/check_drift.py` compares those features with `train.py`'s training split (11,020 rows) using PSI, on 15 monitored features, and says how many entries are real traffic and how many are generated.

**No production traffic has been measured yet.** The API runs on Render's free tier, whose filesystem is wiped on each deploy and restart; point `PREDICTION_LOG_PATH` at a persistent disk before a drift report on real traffic means anything.

What the tooling gives on 1,500 validation listings the models never trained on (`python monitoring/generate_logs.py`, then `python monitoring/check_drift.py monitoring/synthetic_logs.jsonl`). Same population as training, so every feature should read as stable, and does:

| Feature | PSI | Status |
|---|---:|:---|
| Build Year | 0.0239 | ✅ Stable |
| Bedroom Count | 0.0038 | ✅ Stable |
| Livable Surface | 0.0111 | ✅ Stable |
| Total Surface | 0.0287 | ✅ Stable |
| Garage | 0.0000 | ✅ Stable |
| Terrace | 0.0047 | ✅ Stable |
| Swimming Pool | 0.0009 | ✅ Stable |
| Energy Consumption | 0.0076 | ✅ Stable |
| Property State (encoded) | 0.0130 | ✅ Stable |
| Preschool Distance | 0.0111 | ✅ Stable |
| Train Station Distance | 0.0035 | ✅ Stable |
| Supermarket Distance | 0.0122 | ✅ Stable |
| Postcode | 0.0081 | ✅ Stable |
| Area Latitude | 0.0037 | ✅ Stable |
| Area Longitude | 0.0067 | ✅ Stable |

> `property_age` is intentionally **not** monitored: it is `CURRENT_YEAR - build_year`, so its PSI repeats build year's. The list lives in `src/features.py::MONITORED_FEATURES`.

**Reading it:** below 0.1 stable, 0.1–0.25 moderate, above 0.25 strong. An earlier version of this section reported strong drift on 7 of 12 features; that table compared randomly generated values (including a state the model does not know) with an outdated baseline, so it described the generator, not production. See `docs/decisions.md`, 2026-10-12.

---

## Automated Testing

```bash
pip install pytest httpx huggingface_hub
pytest
```

Expected: `128 passed, 1 skipped`, offline (the UI tests mock the API; prediction logging is disabled). The skipped test runs the synthetic log generator and needs `data/`, which is not in git.

- `test_app.py`: health check, valid prediction, invalid postcode (422).
- `test_predict.py`: vocabulary mapping (aliases, idempotence, unknown and rare cities, unknown categories), real-model behaviour (house ≠ apartment, province and condition move the price, the luxury model is not constant), luxury responses without a range, 422 on unknown province, conformal margin reaching its target coverage and applied by the engine, the engine called with the required fields alone pricing like the API, a left-out build year priced at the training median.
- `test_robustness.py`: no fallback routing threshold, no exception text in a 500, `build_year` bounded by today's date, a left-out bedroom count priced at the training median, both quantile models built from one configuration that matches the committed ones.
- `test_monitoring_and_tools.py`: the log appends one line per prediction and loses none under concurrency, carries the features the models read, and feeds a drift report that reads same-population data as stable; the generator never touches the real log; the CLI uses the API's defaults and gives the API's price; a missing range calibration fails instead of a zero margin; the four splits are disjoint, cover every row, and no test row was in the old validation split; target and tiers match what training always used; the search scores the shipped setup.
- `test_routing_blend.py`: the blend measured by `scripts/evaluate_routing_blend.py` is each model at its ends and a log-space mix in between.
- `test_streamlit_app.py`: the UI sends training-vocabulary values, renders the price and range, updates cities with the province, handles luxury, 422 and unreachable-API cases, hides the API docs link where the API is private, and starts the way Streamlit Cloud runs it.
- `test_publish_hf.py`: each Hugging Face repo receives the runtime files and nothing untracked, the Space card matches the port the container serves, `start.sh` keeps Unix line endings, and a dry run uploads nothing.

---

## Future Improvements

- **A softer switch than the hard threshold**, but not the obvious ones: blending both models with the classifier's probability was measured (`scripts/evaluate_routing_blend.py`, validation split, so the test split stays sealed). With the raw probability the €1.95M+ tier improves (40.8% → 26.9% MAPE) but the < €1M tier goes from +7.2% to +16.0% mean bias (26.9% MAPE). With an isotonic-calibrated probability both upper tiers improve (28.1% → 25.2% and 40.8% → 35.6%) but the < €1M tier gets worse (20.6% → 21.1%). That's on 52 listings between €1M and €1.95M and 19 above, and half of them were used to fit the calibration. On the half it never saw (25 and 7 listings), the middle tier goes from 28.5% to 26.1%, the top tier gets worse (34.6% → 36.8%), and so does the < €1M tier (20.5% → 20.9%). The 0.9 threshold stays.
- **Persist the prediction log** on Render (a disk behind `PREDICTION_LOG_PATH`), then read the first drift report on real traffic.
- **Refresh the data**: one scrape, no re-scrape pipeline.

---

## Technologies Used

XGBoost · scikit-learn · category_encoders · FastAPI · Streamlit · Pydantic · pytest · Docker · PSI monitoring · Joblib · Python

---

## Conclusion

This project turns a trained price model into a working service. An XGBoost model handles most predictions, while a second model and a routing classifier take over for luxury properties, using a threshold picked after testing different price tiers. Each prediction now comes with a range (not just one number), so users can see how uncertain the estimate is.

Along the way, several real bugs were found and fixed: a mismatch between a stored price and a probability that quietly turned off the luxury model, an indexing bug that mixed up rows after splitting the data, and a test set that overlapped too much with the training data, making early results look better than they really were. A later end-to-end audit of the deployed service found more: the API spelled categories differently from the training data (so the model ignored property type and province), the luxury model always answered €1.9M, and the evaluation "holdout" still contained most training rows. With those fixed, every model reproducible from `train.py`, and the figures taken on a test split nothing else touched, routing at 0.9 as deployed gives 20.6% average error, 12.9% median error and 44.0% above €1.95M. Six of those points came in one day: the models had never read the energy consumption or the postcode, and three listings under €10,000 were not prices.

The monitoring tools complete the picture. Drift detection checks whether new data still looks like the training data (it reads same-population data as stable; no production traffic has been logged durably yet), and the evaluation script tracks whether the routing decision is still working well over time. This tool does not replace a real estate expert (the app says so clearly), but it gives a solid base to keep the model useful and trustworthy after deployment.

---

## Author

**Siegried Camus**

Developed as part of the **BeCode AI & Data Science Bootcamp**.
