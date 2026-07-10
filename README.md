# 🏠 Immo Eliza Deployment

## 📖 Mission

This project is the deployment phase of the **Immo Eliza Machine Learning** project.

The objective is to make a trained machine learning model available through a REST API and a user-friendly web interface. The application predicts the selling price of residential properties in Belgium using property characteristics provided by the user.

The deployment includes:

* 🚀 A **FastAPI** backend exposing a prediction endpoint.
* 🎨 A **Streamlit** frontend for interactive predictions.
* 🤖 A trained **XGBoost** regression model.
* ✅ Automated API tests using **pytest**.
* 📊 Prediction logging and data monitoring.
* 📈 Drift detection using the Population Stability Index (PSI).

---

# 🛠️ Project Architecture

```
immo-eliza-deployment/
├── api/
│   ├── Dockerfile
│   ├── __init__.py
│   ├── app.py
│   └── predict.py
├── data/
│   ├── clean/
│   │   └── cleaned_data.json
│   └── training_baseline.csv
├── models/
│   ├── best_XGBoost.json
│   ├── preprocessor.joblib
│   └── pipeline.joblib
├── monitoring/
│   ├── __init__.py
│   ├── check_drift.py
│   ├── generate_logs.py
│   ├── metrics.py
│   ├── monitor.py
│   └── logs.json
├── scripts/
│   ├── evaluate.py
│   └── predict_cli.py
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── train.py
├── streamlit/
│   ├── Dockerfile
│   ├── __init__.py
│   └── app.py
├── tests/
│   ├── prediction1.png
│   ├── prediction2.png
│   ├── test_app.py
│   └── UptimeRobot.png
├── .gitignore
├── docker-compose.yml
└── requirements.txt
```

---

# 📂 File Purpose

- **api/app.py**: Defines the FastAPI application, endpoints for predictions and health checks, and validates incoming request data. The `/ping` endpoint acts as a keep-alive monitor used by external services (UptimeRobot) to ensure the API remains active and does not fall into a sleep state.
- **api/predict.py**: Manages the prediction engine by loading the combined `pipeline.joblib` artifact (preprocessor + model) and executing inference.
- **monitoring/monitor.py**: Contains utility functions to log prediction data (`log_prediction`) and detect data drift using the Population Stability Index (`detect_drift`).
- **monitoring/check_drift.py**: Compares logged production predictions (`monitoring/logs.json`) against the training baseline (`data/training_baseline.csv`) and prints a per-feature PSI drift report.
- **monitoring/metrics.py**: Standalone evaluation script that loads `models/pipeline.joblib` and scores it against `data/clean/cleaned_data.json`, reporting MAE, RMSE, MAPE, a bias direction (over/under-estimation), and the 10 worst individual prediction errors — useful for spotting where the model struggles most.
- **monitoring/generate_logs.py**: Generates synthetic, realistic property data (8,000 samples by default) and populates `monitoring/logs.json` to test the monitoring/drift pipeline without needing real production traffic.
- **src/features.py**: Provides the single feature engineering pipeline (`add_features`) used to clean and format data consistently for both training and inference. **This is the source of truth** — every other script (`train.py`, `evaluate.py`, `predict_cli.py`) imports from here to avoid train/inference skew.
- **src/train.py**: Handles the full training workflow: loading and outlier-capping the data, feature engineering, fitting the preprocessor + XGBoost model, and saving three artifacts — the raw XGBoost model (`best_XGBoost.json`), the fitted preprocessor (`preprocessor.joblib`), and the combined deployable pipeline (`pipeline.joblib`). Also writes `data/training_baseline.csv`, the reference distribution used by the drift detector.
- **scripts/evaluate.py**: Loads `pipeline.joblib` and scores it against `data/clean/cleaned_data.json`, printing MAE, RMSE, and MAPE.
- **scripts/predict_cli.py**: Interactive command-line interface that lets you manually input property characteristics and get an instant price estimate from the trained pipeline.
- **streamlit/app.py**: Builds the web interface for property price predictions and sends user inputs to the API.
- **streamlit/Dockerfile**: Configures the container environment specifically for the Streamlit web application.
- **tests/test_app.py**: Automated tests verifying API health and prediction correctness.
- **docker-compose.yml**: Defines and orchestrates the multi-container setup for running both the API and the Streamlit application together.

---

# 🚀 Running the Application

You can run the application using two methods: either manually for development, or using Docker for a production-like environment.

## Method 1: Using Docker Compose (Recommended)

Docker Compose allows you to run both the FastAPI backend and the Streamlit interface simultaneously in isolated, consistent environments.

### What is Docker Compose?

Docker Compose is a tool used to define and run multi-container Docker applications. It reads the `docker-compose.yml` file to orchestrate the services, ensuring that the API and web interface can communicate seamlessly without manual configuration, regardless of your local machine's setup.

### How to launch it

Ensure you are at the root of the `immo-eliza-deployment/` folder, then run:

```bash
docker compose up --build
```

Docker will automatically build your containers, install dependencies, and start both services. Once running, you can access the interface at:
**[http://localhost:8501](http://localhost:8501)**

---

## Method 2: Manual Execution (Development)

If you prefer to run services directly on your host machine:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the application

You will need to open two separate terminal windows:

**Terminal 1 (API):**

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 (Streamlit):**

```bash
streamlit run streamlit/app.py
```

Open your browser at:
**http://localhost:8501**

---

# 🛡️ Reliability & Uptime Monitoring

To ensure the application remains stable and available 24/7, this project uses three layers of reliability:

- **Docker Orchestration**: Docker manages the lifecycle of your containers. If the API or Streamlit service crashes due to an unexpected error, Docker automatically restarts the affected container, ensuring minimal downtime.
- **A dedicated `/ping` endpoint**: The FastAPI app exposes a lightweight `/ping` route that does nothing but confirm the service is alive. It's the hook that any external monitor uses to check the app is awake — cheap enough to call every few minutes without adding real load.
- **External Monitoring with UptimeRobot**: Cloud platforms often put free-tier services into "sleep mode" after a period of inactivity. UptimeRobot is configured to send periodic requests to the `/ping` endpoint. This keeps the service active, prevents runtime timeouts, and guarantees the model is ready to predict whenever a user visits the interface. A screenshot of the live UptimeRobot monitor is available at `tests/UptimeRobot.png` as proof the ping has been running successfully.

**Note on monitoring**: with Docker, services automatically restart on crash. On the cloud, the `/ping` endpoint is what UptimeRobot polls to keep the app awake and responsive at all times.

---

# 🏡 Using the Application

The user fills in the property characteristics, including:

* Property type
* Location
* Living surface
* Number of bedrooms
* Construction year
* Energy consumption
* Outdoor features
* Distances to nearby facilities

After clicking **Predict Price**, the application sends the request to the API and returns the estimated selling price.

---

# 🔮 Prediction Examples

The screenshots below show two examples of price predictions generated by the deployed Streamlit application.

## 🏢 Example 1 – Standard Apartment

![Prediction Example 1](tests/prediction1.png)

*Figure 1.* Prediction for a standard apartment located in Brussels. The application displays both the input characteristics and the estimated market value generated by the XGBoost model.

---

## 🏡 Example 2 – Detached House

![Prediction Example 2](tests/prediction2.png)

*Figure 2.* Prediction for a detached house with premium features. This example demonstrates the model's ability to estimate prices for larger and more valuable residential properties.

---

# 📊 Model Performance & Monitoring

## Model Performance

Evaluated with `monitoring/metrics.py` against 15,746 records from `data/clean/cleaned_data.json`:

| Metric | Value |
| :--- | :--- |
| **Total Records Evaluated** | 15,746 |
| **Mean Absolute Error (MAE)** | €60,831.85 |
| **Root Mean Squared Error (RMSE)** | €242,698.80 |
| **Mean Absolute Percentage Error (MAPE)** | 16.24% |
| **Bias** | Overestimating prices by 5.64% on average |

*Analysis:* the large gap between MAE (~€61k) and RMSE (~€243k) points to a small number of very large errors dragging the average up, rather than the model being uniformly imprecise. The 3 worst errors make this obvious — all three are ultra-luxury properties priced above €8M, where the model's prediction is off by €7M–€8.8M:

| Actual | Predicted | Error |
| :--- | :--- | :--- |
| €8,000,000 | €833,476 | €7,166,524 |
| €8,900,000 | €940,093 | €7,959,907 |
| €8,994,000 | €175,950 | €8,818,050 |

**Why these results, and why removing the hard cap changed nothing:**
The €3,000,000 hard cap in `src/train.py` was removed, but the reported metrics are *identical* to the previous run (same MAE, RMSE, MAPE, same worst errors, down to the cents). That's the tell: the hard cap was never actually the binding constraint. `load_data()` also trims the dataset to the **1st–99th percentile** of `price` before the hard cap is even applied, and `clean_target()` clips the target again with the same percentile logic right before training. Since the 99th percentile of the price distribution already sits well below €3M, that quantile-based trim was doing all the work — removing the €3,000,000 ceiling was redundant, not the fix.

In other words: the model still has **never seen a property above roughly the 99th percentile of the training set**, regardless of the hard cap. Ultra-luxury properties (€5M+) are structurally out-of-distribution, so predicting €175k–€1.6M for an €8-9M mansion isn't a bug — it's the model correctly reproducing patterns learned from the mass market and having no reference point for anything above it.

**Recommendation**: if the goal is to also cover the luxury segment, the percentile-based trimming in `load_data()`/`clean_target()` needs to be loosened (or replaced with an explicit ceiling that's actually higher than the natural 99th percentile), not just the separate hard cap. Otherwise, report "in-distribution" performance (sub-99th-percentile) separately from full-dataset performance, since blending them makes the headline MAPE look far worse than the model's real behavior on typical properties.

**Tested and rejected**: widening `QUANTILE_UPPER` from 0.99 to 0.999 (letting more luxury properties into training) was tried and made things *worse* across the board — MAPE rose from 16.24% to 16.54%, and bias from 5.64% to 6.89% overestimation — without meaningfully fixing the luxury predictions themselves. There simply aren't enough high-end properties in the dataset for the model to learn a reliable pattern from them; a few extra examples just added noisy, high-leverage signal that hurt the mass-market segment too. **The metrics in this README reflect the original 0.99 bound**, which remains the best-performing configuration found so far. Reliably predicting the luxury segment would need either substantially more high-end training examples, or a dedicated model trained specifically for that segment.

## 🚀 Future Improvements

* **Stratify evaluation by price tier.** Report MAE/RMSE/MAPE separately for e.g. `<€1M`, `€1M–€3M`, `€3M+` instead of one blended number, so a handful of luxury outliers can't dominate the headline metric.
* **Train a dedicated luxury-segment model, rather than widening the clip.** Confirmed by testing: simply loosening `QUANTILE_UPPER` doesn't work — there isn't enough high-end data for the main model to learn from. A separate model (or at minimum, gathering significantly more luxury listings) would be needed.
* **Add prediction intervals, not just point estimates.** Quantile regression or a conformal prediction wrapper around the XGBoost model would let the app show a price *range* and flag low-confidence predictions instead of a single number that looks equally confident everywhere.
* **Retrain against the drifted features.** The drift report shows the strongest shifts in `property_state_encoded`, `energy_consumption`, and location-distance features — retraining on more recent data should prioritize those.
* **Trim the monitored feature list.** Drop `property_age` (redundant with `build_year`) and stop treating `price_per_m2` PSI as a model-input signal, since it no longer feeds the model (see Drift Analysis below).
* **Confirm the evaluation set's provenance.** Check whether `cleaned_data.json` used by `metrics.py`/`evaluate.py` is a held-out test set or overlaps with training data — that materially changes how the 16.24% MAPE should be interpreted.

## ⚠️ Known Limitations & Disclaimer

This model should be treated as an **estimation tool**, not a valuation. Two things a user should be told before trusting the number:

* **Systematic bias**: the model overestimates prices by ~5.64% on average across the evaluated dataset. A predicted price should be read as "likely a bit above the model's honest estimate," not as a precise figure.
* **Out-of-distribution risk on high-end properties**: for anything priced above roughly the 99th percentile of the training data (in practice, multi-million-euro properties), the model is extrapolating far outside what it has ever learned from, and errors of several million euros are possible (see the worst-error table above). The bias above (~5.6%) does **not** apply to this segment — the real error there is much larger and in the *opposite* direction (underestimation).

**Recommendation — surface this in two places:**

1. **In this README**, as done here, so anyone evaluating or maintaining the project understands the model's real accuracy profile.
2. **In the Streamlit app itself**, right next to the predicted price — a short, non-alarming note is enough, e.g.:

   > *"This estimate is generated by a machine learning model and may differ from the actual market value, particularly for atypical or very high-end properties. Historically, predictions have tended to run about 5–6% above the eventual value on average."*

   If `livable_surface`, `price_per_m2` (if reintroduced), or the predicted price itself falls far outside the typical training range, the UI could optionally show a stronger warning (e.g. "this property is outside the range the model was trained on — treat this estimate with caution") rather than presenting every prediction with the same apparent confidence.

## 📉 Drift Analysis

Monitoring was performed by comparing **8,000 live (synthetic) predictions** with the **10,802 samples** used to train the model, using `monitoring/check_drift.py`.
The **Population Stability Index (PSI)** was used to determine whether the production data still follows the same distribution as the original training dataset.

## 🚨 Current Status

**Strong Drift Detected**

| Feature | PSI | Status |
| ---------------------- | ------- | --------------- |
| Build Year | 0.9734 | 🚨 Strong Drift |
| Bedroom Count | 0.2005 | ⚠️ Moderate |
| Livable Surface | 0.6352 | 🚨 Strong Drift |
| Total Surface | 0.0824 | ✅ Stable |
| Garage | 0.1500 | ⚠️ Moderate |
| Terrace | 0.1228 | ⚠️ Moderate |
| Swimming Pool | 0.0009 | ✅ Stable |
| Energy Consumption | 11.7938 | 🚨 Strong Drift |
| Property State (encoded) | 13.5164 | 🚨 Strong Drift |
| Property Age | 0.9734 | 🚨 Strong Drift |
| Preschool Distance | 0.5542 | 🚨 Strong Drift |
| Train Station Distance | 3.0964 | 🚨 Strong Drift |
| Supermarket Distance | 2.1613 | 🚨 Strong Drift |
| Price per m² | 0.3889 | 🚨 Strong Drift |

## 📈 Monitoring Interpretation

The monitoring report reveals that most of the important variables have changed significantly since the model was trained.

The strongest distribution shifts concern:

* 🏷️ Property state (encoded) — the highest PSI of all (13.52)
* ⚡ Energy consumption
* 🚉 Distance to train stations and supermarkets
* 🏗️ Build year / property age
* 📐 Livable surface

These features are among the most influential variables used by the XGBoost model. Since their distributions have evolved considerably, the production data no longer fully represents the original training data.

Two things are worth flagging about the monitored feature list itself:

1. **`build_year` and `property_age` always report the exact same PSI (0.9734)** — expected, since `property_age = 2026 - build_year` is a pure linear transform of the same underlying signal. Monitoring both is redundant; one can safely be dropped from the drift report without losing information.
2. **`price_per_m2` is still being monitored for drift, even though it was explicitly removed from the model's training features** (see the comment in `src/train.py`: it was dropped due to data leakage, since it's derived from the target and is unavailable — effectively always 0 — at real inference time). Its PSI here mostly reflects that production logs and training data populate this field differently, not that the model itself is affected. It's safe to keep as an informational signal, but it shouldn't be read as evidence that a *model input* has drifted.

Although the prediction service remains operational and continues to return valid property price estimates, the observed drift indicates that prediction accuracy may gradually decrease over time.

The monitoring component therefore recommends retraining the model using more recent real estate data before the next production deployment.

---

# 🧪 Automated Testing

Automated tests verify that both the API and the prediction pipeline work correctly.

## Install test dependencies

```bash
pip install pytest httpx
```

## Run the tests

```bash
pytest tests/test_app.py
```

Expected output:

```text
tests/test_app.py .... [100%]
================== 4 passed ==================
```

## ✔️ What is tested?

* ❤️ API availability through the health endpoint.
* 🏠 Successful prediction using valid real estate data.
* 💰 Correct numeric prediction returned by the model.
* ❌ Validation of incorrect input types.
* ⚠️ Detection of missing required fields (HTTP 422).

Passing all tests confirms that the deployed application behaves as expected.

---

# 📚 Technologies Used

| Component | Technology |
| ---------------- | -------------------------------- |
| Machine Learning | XGBoost |
| API | FastAPI |
| Frontend | Streamlit |
| Data Validation | Pydantic |
| Testing | pytest |
| Monitoring | Population Stability Index (PSI) |
| Serialization | Joblib |
| Language | Python |

---

# 📌 Final Conclusion

This project demonstrates the complete deployment lifecycle of a machine learning regression model.

A trained **XGBoost** model was packaged behind a **FastAPI** REST API and connected to an interactive **Streamlit** web application, allowing users to estimate Belgian real estate prices in real time.

Beyond simple deployment, the project integrates several software engineering and MLOps practices designed to improve reliability and maintainability:

* ✅ REST API deployment with FastAPI
* ✅ Interactive web interface with Streamlit
* ✅ Automated testing with pytest
* ✅ Input validation using Pydantic
* ✅ Logging of production predictions
* ✅ A dedicated `/ping` health endpoint, polled by UptimeRobot to keep the service awake
* ✅ Population Stability Index (PSI) monitoring
* ✅ Automatic drift detection
* ✅ Retraining recommendation based on monitoring results

Re-running the evaluation and drift scripts surfaced two useful, honest findings rather than just a clean report card:

1. The headline error metrics are dominated by a handful of ultra-luxury properties (€5M+) that fall outside the €3M training cap — the model isn't broken, it's simply being asked to extrapolate far past what it was trained on. Segmenting the evaluation (or the model itself) by price tier would give a much more representative picture of real-world accuracy.
2. The drift report currently monitors two redundant features (`build_year` / `property_age`) and one feature that no longer feeds the model at all (`price_per_m2`, dropped for data leakage). Neither is harmful, but trimming the monitored feature list to only the model's actual inputs would make the PSI report more directly actionable.

This project illustrates that deploying a machine learning model is not the end of the workflow. Continuous monitoring, automated testing, data validation, and periodic retraining are essential components of a reliable production machine learning system — and that monitoring output is only useful if it's periodically sanity-checked against what the model was actually trained to do.

Overall, the application provides a complete end-to-end deployment pipeline, from model serving and user interaction to monitoring and maintenance, following modern software engineering and MLOps best practices.

## Author

**Siegried Camus**
Immo Eliza Deployment project developed as part of the BeCode AI & Data Science Bootcamp.
