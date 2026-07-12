# 🏠 Immo Eliza Deployment

Deployment phase of the **Immo Eliza Machine Learning** project: a REST API + web interface that predicts Belgian residential property prices, with a dedicated luxury-segment model for high-end properties.

- 🚀 FastAPI backend + 🎨 Streamlit frontend
- 🤖 XGBoost regression, with a luxury-segment model + routing classifier
- ✅ Automated pytest suite
- 📊 Prediction logging, PSI-based drift detection

---

## 📑 Table of Contents

1. [Project Architecture](#-project-architecture)
2. [File Purpose](#-file-purpose)
3. [Running the Application](#-running-the-application)
4. [Reliability & Uptime Monitoring](#-reliability--uptime-monitoring)
5. [Using the Application](#-using-the-application)
6. [Prediction Examples](#-prediction-examples)
7. [Model Performance & Monitoring](#-model-performance--monitoring)
8. [Luxury Routing](#-luxury-routing)
9. [Known Limitations & Disclaimer](#-known-limitations--disclaimer)
10. [Drift Analysis](#-drift-analysis)
11. [Automated Testing](#-automated-testing)
12. [Future Improvements](#-future-improvements)
13. [Technologies Used](#-technologies-used)
14. [Conclusion](#-conclusion)

---

## 🛠️ Project Architecture

```text
immo-eliza-deployment/
├── api/
│   ├── __init__.py
│   ├── app.py               # FastAPI app, request validation, luxury routing
│   ├── Dockerfile
│   └── predict.py            # Loads model artifacts, runs inference
├── data/
├── models/
├── monitoring/
│   ├── __init__.py
│   ├── check_drift.py        # PSI drift report vs. training baseline
│   ├── generate_logs.py       # Synthetic logs for testing the monitoring pipeline
│   ├── metrics.py             # True-holdout evaluation, by price tier
│   └── monitor.py             # log_prediction(), PSI computation
├── scripts/
│   ├── evaluate_stratified.py  # Sweeps the luxury-routing threshold, per tier
│   └── predict_cli.py
├── src/
│   ├── __init__.py
│   ├── features.py            # Single source of truth for feature engineering
│   ├── optimize.py            # Optuna hyperparameter search
│   └── train.py               # Full training workflow (standard + luxury + routing + quantile models)
├── streamlit/
│   ├── __init__.py
│   ├── app.py
│   └── Dockerfile
├── tests/
│   ├── __pycache__/
│   ├── prediction1.png
│   ├── prediction2.png
│   ├── test_app.py
│   └── uptimeRobot.png
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── README.md
└── requirements.txt
```

> `scripts/evaluate_stratified.py` reports metrics **per price tier** and sweeps the luxury-routing threshold — replaces the old global-metric `evaluate.py`.

---

## 📂 File Purpose

| File | Purpose |
|---|---|
| `api/app.py` | FastAPI endpoints, request validation, luxury routing decision (via `luxury_threshold.json`), `/ping` keep-alive for UptimeRobot |
| `api/predict.py` | Loads the 5 model artifacts (standard, luxury, luxury classifier, lower/upper quantile) and runs inference |
| `monitoring/monitor.py` | Logs predictions, computes PSI drift |
| `monitoring/check_drift.py` | Compares logs vs. training baseline, prints PSI report |
| `monitoring/metrics.py` | Scores `pipeline.joblib` on a **true holdout** (excludes `training_baseline.csv` rows) |
| `monitoring/generate_logs.py` | Generates synthetic logs to test the monitoring pipeline |
| `src/features.py` | Single source of truth for feature engineering — imported by every script to prevent train/inference skew |
| `src/train.py` | Full training workflow: cleans data, caps outliers, trains standard model, luxury model, routing classifier, and quantile (interval) models; writes `luxury_threshold.json` and `training_baseline.csv` |
| `src/optimize.py` | Optuna hyperparameter search for the standard model |
| `scripts/evaluate_stratified.py` | Evaluates the full routing pipeline per price tier across routing thresholds — used to tune the 0.7 threshold |
| `scripts/predict_cli.py` | CLI for manual predictions |
| `streamlit/app.py` | Web UI, sends inputs to the API, shows uncertainty disclaimer + luxury routing flag |
| `tests/test_app.py` | API + prediction automated tests |
| `docker-compose.yml` | Orchestrates API + Streamlit containers |

> **Legacy note:** `pipeline.joblib` now bundles preprocessor + model in one `sklearn.Pipeline`, replacing the old separate `best_XGBoost.json` / `preprocessor.joblib` files.

---

## 🚀 Running the Application

### 🌐 Live Deployment

- **Web App (Streamlit Community Cloud):** https://immo-eliza-deployment-sieg.streamlit.app
- **API (Render):** https://immo-eliza-a.onrender.com
- **API Docs (Swagger UI):** https://immo-eliza-ui.onrender.com/docs

### Method 1 — Docker Compose (recommended)

```bash
docker compose up --build
```

App available at `http://localhost:8501`.

### Method 2 — Manual (development)

```bash
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000     # terminal 1
streamlit run streamlit/app.py                      # terminal 2
```

### Method 3 — Retrain & re-evaluate

```bash
python src/train.py                    # trains all 3 artifacts
python scripts/evaluate_stratified.py  # sweeps luxury-routing threshold
python monitoring/metrics.py           # true-holdout metrics by tier
```

---

## 🛡️ Reliability & Uptime Monitoring

- **Docker** auto-restarts either container on crash.
- **`/ping`** endpoint: lightweight liveness check.
- **UptimeRobot** polls `/ping` to prevent cold starts on free-tier hosting (see `tests/UptimeRobot.png`).

---

## 🏡 Using the Application

Enter property type, location, surface, bedrooms, construction year, energy consumption, outdoor features, and distances to amenities. Streamlit sends the request to FastAPI and displays the estimated price, flagging if it was routed to the luxury model.

---

## 🔮 Prediction Examples

The screenshots below show two real predictions from the deployed Streamlit app.

**Example 1 — Standard property:**

![Prediction 1](tests/prediction1.png)

Estimated price: **€1,310,313**

**Example 2 — Smaller property:**

![Prediction 2](tests/prediction2.png)

Estimated price: **€212,262**

---

## 📊 Model Performance & Monitoring

Evaluated with `monitoring/metrics.py` on a **true holdout** of 4,944 records (rows absent from `training_baseline.csv` — earlier reports were inflated by data leakage from overlapping training rows).

| Metric | Value |
|---|---:|
| MAE | €75,422 |
| RMSE | €337,970 |
| MAPE | 16.83% |
| Bias | +6.30% (overestimation) |

**Prediction interval:** every standard-segment prediction now ships with an 80% interval (10th–90th percentile), from two extra `XGBRegressor` models trained with `objective="reg:quantileerror"` alongside the point-estimate model. Returned in the API response as `prediction_interval: {lower, upper}` and shown in the Streamlit UI. Not yet available for the luxury segment (too few rows for a stable quantile fit).

By tier:

| Tier | N | MAE | MAPE |
|---|---:|---:|---:|
| < €1M | 4,617 | €33,501 | 16.36% |
| €1M–€3M | 298 | €409,672 | 19.74% |
| €3M+ | 29 | €3,314,774 | 62.35% |

The MAE/RMSE gap reflects a handful of extreme errors on ultra-luxury (€8M+) properties, far outside the training distribution — e.g. one €8.99M property was underestimated by €8.77M.

**Widening the outlier clip (0.99 → 0.999 percentile) was tested and rejected:** MAPE worsened (16.24% → 16.54%), confirming the fix needed was a dedicated luxury model, not a looser cap. The standard model also never saw €5M+ properties during training since prices are clipped to the 1st–99th percentile.

---

## 🏛️ Luxury Routing

A dedicated luxury regressor (trained on the top 1% of prices) plus a routing classifier decide, per property, whether to use the standard or luxury model. Threshold tuned via `scripts/evaluate_stratified.py` sweeping 0.4→0.95; **0.7** selected.

| Tier | N | Routed to luxury | MAPE | Bias |
|---|---:|---:|---:|---:|
| < €1M | 4,617 | 0.6% | 18.9% | +10.8% |
| €1M–€3M | 298 | 57.0% | 21.2% | -10.0% |
| €3M+ | 29 | 93.1% | 59.3% | -59.3% |

**Honest takeaway:** routing gives a clear win only on €3M+ (62.35% → 59.3% MAPE); it's roughly neutral-to-slightly-negative below that, since a few misrouted properties skew a tier average more than correct routing improves it. 0.7 was chosen because it keeps 93% of true €3M+ properties correctly routed while limiting `<€1M` false positives to 0.6%.

---

## ⚠️ Known Limitations & Disclaimer

This is an **estimation tool, not a valuation system**.

- **Bias:** overestimates prices by ~6.30% on average.
- **High-end properties (~€3M+):** even the luxury model extrapolates from very few examples; multi-million-euro errors remain possible, typically underestimation.
- **Routing trade-off:** helps €3M+, roughly neutral for lower tiers (see [Luxury Routing](#-luxury-routing)).

These limitations are documented here and surfaced in the Streamlit app as a disclaimer next to each prediction, plus a flag when routed to the luxury model.

---

## 📉 Drift Analysis

`monitoring/check_drift.py` compares 8,000 synthetic production predictions against the 10,802-sample training set using PSI, on 12 monitored features.

| Feature | PSI | Status |
|---|---:|:---|
| Build Year | 0.9737 | 🚨 Strong |
| Bedroom Count | 0.2005 | ⚠️ Moderate |
| Livable Surface | 0.6399 | 🚨 Strong |
| Total Surface | 0.0824 | ✅ Stable |
| Garage | 0.1415 | ⚠️ Moderate |
| Terrace | 0.1049 | ⚠️ Moderate |
| Swimming Pool | 0.0003 | ✅ Stable |
| Energy Consumption | 11.7287 | 🚨 Strong |
| Property State (encoded) | 13.3866 | 🚨 Strong |
| Preschool Distance | 0.5430 | 🚨 Strong |
| Train Station Distance | 3.0922 | 🚨 Strong |
| Supermarket Distance | 2.1685 | 🚨 Strong |

> `property_age` and `price_per_m2` are intentionally **not** monitored: `property_age` is a deterministic function of `build_year` (same PSI, redundant signal), and `price_per_m2` was excluded from the model's training features entirely (to avoid target leakage), so it isn't a meaningful drift signal for this model.

**Interpretation:** 7 of 12 monitored features show strong drift (property state, energy consumption, all three distance features, build year, livable surface), so production data no longer matches the training distribution well. The service still runs and returns valid predictions, but retraining on more recent data is recommended.

---

## 🧪 Automated Testing

```bash
pip install pytest httpx
pytest tests/test_app.py
```

Expected: `4 passed`. Covers: API health check, successful prediction on valid input, numeric correctness, invalid-input validation (422).

---

## 🚀 Future Improvements

| Improvement | Status |
|---|---|
| Stratify evaluation by price tier | ✅ Done |
| Dedicated luxury-property model | ✅ Done (clear benefit only on €3M+) |
| True holdout evaluation | ✅ Done |
| Prediction intervals (quantile/conformal) | ✅ Done |
| Simplify monitored features (drop `property_age`, relabel `price_per_m2`) | ✅ Done |

---

## 📚 Technologies Used

XGBoost · FastAPI · Streamlit · Pydantic · pytest · PSI monitoring · Joblib · Python

---

## 📌 Conclusion

This project turns a trained price model into a working service. An XGBoost model handles most predictions, while a second model and a routing classifier take over for luxury properties, using a threshold picked after testing different price tiers. Each prediction now comes with a range (not just one number), so users can see how uncertain the estimate is.

Along the way, several real bugs were found and fixed: a mismatch between a stored price and a probability that quietly turned off the luxury model, an indexing bug that mixed up rows after splitting the data, and a test set that overlapped too much with the training data, making early results look better than they really were. The final numbers are reported honestly: 16.83% average error overall, rising to 62% for the hardest €3M+ properties.

The monitoring tools complete the picture. Drift detection checks whether new data still looks like the training data (right now, 7 out of 12 features show strong drift), and the evaluation script tracks whether the routing decision is still working well over time. This tool does not replace a real estate expert — the app says so clearly — but it gives a solid base to keep the model useful and trustworthy after deployment.

---

## 👤 Author

**Siegried Camus**

Developed as part of the **BeCode AI & Data Science Bootcamp**.