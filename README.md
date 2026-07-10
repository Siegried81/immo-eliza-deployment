# 🏠 Immo Eliza Deployment

## 📖 Mission

This project is the deployment phase of the **Immo Eliza Machine Learning** project.

The objective is to make a trained machine learning model available through a REST API and a user-friendly web interface.

The application predicts the selling price of residential properties in Belgium using property characteristics provided by the user.

The deployment includes:

- 🚀 A FastAPI backend exposing a prediction endpoint.
- 🎨 A Streamlit frontend for interactive predictions.
- 🤖 A trained XGBoost regression model.
- ✅ Automated API tests using **pytest**.
- 📊 Prediction logging and data monitoring.
- 📈 Drift detection using the **Population Stability Index (PSI)**.

---

# 🛠️ Project Architecture

```text
immo-eliza-deployment/
├── api/
│   ├── Dockerfile
│   ├── __init__.py
│   ├── app.py
│   └── predict.py
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

### `api/app.py`

Defines the FastAPI application, endpoints for predictions and health checks, and validates incoming request data.

The `/ping` endpoint acts as a keep-alive monitor used by external services (UptimeRobot) to ensure the API remains active and does not fall into a sleep state.

### `api/predict.py`

Manages the prediction engine by loading the combined `pipeline.joblib` artifact (preprocessor + model) and executing inference.

### `monitoring/monitor.py`

Contains utility functions to:

- Log prediction data (`log_prediction`)
- Detect data drift using the Population Stability Index (`detect_drift`)

### `monitoring/check_drift.py`

Compares logged production predictions (`monitoring/logs.json`) against the training baseline (`data/training_baseline.csv`) and prints a per-feature PSI drift report.

### `monitoring/metrics.py`

Standalone evaluation script that loads `models/pipeline.joblib` and scores it against `data/clean/cleaned_data.json`, reporting:

- MAE
- RMSE
- MAPE
- Bias direction (over/under-estimation)
- The 10 worst individual prediction errors

Useful for spotting where the model struggles the most.

### `monitoring/generate_logs.py`

Generates synthetic, realistic property data (8,000 samples by default) and populates `monitoring/logs.json` to test the monitoring and drift pipeline without requiring real production traffic.

### `src/features.py`

Provides the single feature engineering pipeline (`add_features`) used to clean and format data consistently for both training and inference.

This is the source of truth: every other script (`train.py`, `evaluate.py`, `predict_cli.py`) imports this module to avoid train/inference skew.

### `src/train.py`

Handles the complete training workflow:

- Loads the cleaned dataset
- Caps outliers
- Applies feature engineering
- Fits the preprocessor and XGBoost model
- Saves three deployment artifacts:
  - `best_XGBoost.json`
  - `preprocessor.joblib`
  - `pipeline.joblib`

It also creates `data/training_baseline.csv`, which serves as the reference dataset for drift detection.

### `scripts/evaluate.py`

Loads `pipeline.joblib` and evaluates the model against `data/clean/cleaned_data.json`, reporting:

- MAE
- RMSE
- MAPE

### `scripts/predict_cli.py`

Interactive command-line interface allowing users to manually enter property characteristics and instantly obtain a price prediction from the trained pipeline.

### `streamlit/app.py`

Builds the Streamlit web interface and sends user inputs to the FastAPI backend for prediction.

### `streamlit/Dockerfile`

Defines the container environment specifically for the Streamlit application.

### `tests/test_app.py`

Contains automated tests verifying API availability and prediction correctness.

### `docker-compose.yml`

Defines and orchestrates the multi-container setup running both the FastAPI backend and the Streamlit frontend.

---

# 🚀 Running the Application

The application can be launched either manually (development mode) or using Docker (recommended).

## Method 1 — Docker Compose (Recommended)

Docker Compose runs both the FastAPI backend and the Streamlit frontend inside isolated containers.

### What is Docker Compose?

Docker Compose is a tool used to define and run multi-container Docker applications.

It reads the `docker-compose.yml` file and automatically starts all required services while ensuring they can communicate with each other.

### Launch the application

From the root directory of the project, run:

```bash
docker compose up --build
```

Docker will:

- build the containers,
- install all dependencies,
- launch both services.

The application will then be available at:

```
http://localhost:8501
```

---

## Method 2 — Manual Execution (Development)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API

Open a first terminal and run:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 3. Start Streamlit

Open a second terminal and run:

```bash
streamlit run streamlit/app.py
```

Open your browser at:

```
http://localhost:8501
```

---

# 🛡️ Reliability & Uptime Monitoring

To ensure the application remains stable and available 24/7, the project uses three complementary reliability mechanisms.

### 🐳 Docker Orchestration

Docker manages the lifecycle of both containers.

If either the API or the Streamlit application crashes unexpectedly, Docker automatically restarts the affected container, minimizing downtime.

### ❤️ `/ping` Health Endpoint

The FastAPI application exposes a lightweight `/ping` endpoint whose only purpose is to confirm that the API is alive.

External monitoring services can call this endpoint every few minutes without placing any significant load on the application.

### 🌐 External Monitoring with UptimeRobot

Cloud providers often put free-tier applications to sleep after a period of inactivity.

UptimeRobot periodically sends requests to the `/ping` endpoint to:

- keep the application awake,
- prevent cold starts,
- ensure predictions remain immediately available.

A screenshot of the monitoring dashboard is available in:

```
tests/UptimeRobot.png
```

as proof that the monitoring system is running correctly.

> **Note**
>
> - With Docker, containers automatically restart after a crash.
> - On cloud deployments, the `/ping` endpoint is continuously polled by UptimeRobot to keep the API responsive.

---

# 🏡 Using the Application

The user fills in the property characteristics, including:

- Property type
- Location
- Living surface
- Number of bedrooms
- Construction year
- Energy consumption
- Outdoor features
- Distances to nearby facilities

After clicking **Predict Price**, the Streamlit application sends the request to the FastAPI API, which returns the estimated selling price.

---

# 🔮 Prediction Examples

The screenshots below illustrate two predictions generated by the deployed Streamlit application.

## 🏢 Example 1 – Standard Apartment

*Figure 1.*

Prediction for a standard apartment located in Brussels.

The application displays both the input characteristics and the estimated market value generated by the XGBoost model.

```md
![Prediction Example 1](tests/prediction1.png)
```

---

## 🏡 Example 2 – Detached House

*Figure 2.*

Prediction for a detached house with premium features.

This example demonstrates the model's ability to estimate prices for larger and more valuable residential properties.

```md
![Prediction Example 2](tests/prediction2.png)
```
# 📊 Model Performance & Monitoring

## 📈 Model Performance

The model was evaluated using `monitoring/metrics.py` on **15,746** records from `data/clean/cleaned_data.json`.

| Metric | Value |
|---------|------:|
| Total Records Evaluated | 15,746 |
| Mean Absolute Error (MAE) | **€60,831.85** |
| Root Mean Squared Error (RMSE) | **€242,698.80** |
| Mean Absolute Percentage Error (MAPE) | **16.24%** |
| Bias | Overestimating prices by **5.64%** on average |

---

## 📈 Performance Analysis

The large difference between **MAE (~€61k)** and **RMSE (~€243k)** indicates that the model performs well for most properties, while a very small number of extreme prediction errors significantly increase the RMSE.

These outliers correspond almost exclusively to **ultra-luxury properties** priced above **€8 million**, which lie far outside the distribution seen during training.

The three largest prediction errors are shown below.

| Actual Price | Predicted Price | Absolute Error |
|--------------|----------------:|---------------:|
| €8,000,000 | €833,476 | €7,166,524 |
| €8,900,000 | €940,093 | €7,959,907 |
| €8,994,000 | €175,950 | €8,818,050 |

---

## ❓ Why Removing the Hard Cap Changed Nothing

Removing the €3M hard cap had **no impact** on the evaluation metrics.

The reason is that the real limitation was never the hard cap itself:

- `load_data()` and `clean_target()` already clip prices to the **1st–99th percentile**.
- That percentile lies well below €3M.
- Consequently, the model has never seen €5M+ properties during training.

Without representative examples of luxury properties, the model simply has no basis for extrapolating accurately to this market segment.

---

## 🧪 Tested and Rejected

Increasing

```python
QUANTILE_UPPER = 0.999
```

was tested but produced worse results:

| Metric | Original (0.99) | New (0.999) |
|---------|----------------:|------------:|
| MAPE | **16.24%** | **16.54%** |
| Bias | **5.64%** | **6.89%** |

Luxury predictions remained poor while overall performance degraded.

This confirms that simply widening the clipping threshold is **not** an effective solution.

Reliable predictions for high-end properties would instead require:

- substantially more luxury listings,
- or a dedicated luxury-property model.

---

# 🚀 Future Improvements

Several improvements could further increase the reliability of the deployment.

### 📊 Stratify evaluation by price tier

Instead of reporting a single MAE/RMSE/MAPE value, evaluate the model separately for:

- properties below €1M,
- properties between €1M and €3M,
- luxury properties above €3M.

This prevents a handful of extreme outliers from dominating the global metrics.

---

### 🏡 Train a dedicated luxury-property model

Testing confirmed that simply increasing `QUANTILE_UPPER` does not improve predictions for expensive properties.

A dedicated model trained specifically on luxury listings—or a significantly larger luxury dataset—would likely perform much better.

---

### 📈 Add prediction intervals

Instead of returning only a point estimate, provide a confidence interval using:

- Quantile Regression, or
- Conformal Prediction.

This would allow users to understand the uncertainty associated with each prediction.

---

### 🔄 Retrain using more recent data

The drift analysis indicates significant distribution shifts in several important variables, including:

- `property_state_encoded`
- `energy_consumption`
- distance-related features

Retraining on newer real-estate data should therefore be prioritized.

---

### 🧹 Simplify the monitored features

The monitoring report currently contains redundant information.

Possible improvements include:

- removing `property_age`, which duplicates `build_year`,
- excluding `price_per_m2` from PSI monitoring since it is no longer a model input.

---

### ✅ Verify the evaluation dataset

Confirm whether `cleaned_data.json` represents a true held-out test set or overlaps with the training data.

This distinction has a significant impact on interpreting the reported **16.24% MAPE**.

---

# ⚠️ Known Limitations & Disclaimer

This model should be considered **an estimation tool rather than a property valuation system**.

Users should be aware of two important limitations.

## 📈 Systematic Bias

Across the evaluation dataset, the model tends to **overestimate prices by approximately 5.64%**.

Predictions should therefore be interpreted as approximate estimates rather than precise valuations.

---

## 🏰 High-End Properties

For properties priced above roughly the **99th percentile** of the training data (multi-million-euro properties), the model is forced to extrapolate far beyond the examples it has learned from.

Prediction errors of several million euros are therefore possible.

The average bias reported above **does not apply** to this segment, where errors become much larger and typically correspond to severe underestimation.

---

## 💡 Recommendation

These limitations should be communicated in two places.

### In this README

This documentation helps developers and evaluators understand the model's actual strengths and weaknesses.

### In the Streamlit application

A short note displayed next to each prediction would improve transparency, for example:

> *"This estimate is generated by a machine learning model and may differ from the actual market value, particularly for atypical or very high-end properties. Historically, predictions have tended to run approximately 5–6% above the eventual value on average."*

For properties lying well outside the training distribution, the application could display a stronger warning such as:

> *"This property falls outside the range used during model training. The prediction should therefore be interpreted with caution."*

---

# 📉 Drift Analysis

Monitoring compares **8,000 synthetic production predictions** against the **10,802 samples** used during model training using `monitoring/check_drift.py`.

Data drift is measured using the **Population Stability Index (PSI)**.

---

## 🚨 Current Status

### Strong Drift Detected

| Feature | PSI | Status |
|---------|----:|:------|
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

---

# 📈 Monitoring Interpretation

The monitoring report indicates that several important input variables have shifted substantially since the model was trained.

The strongest changes concern:

- 🏷️ Property state (encoded)
- ⚡ Energy consumption
- 🚉 Distance to train stations
- 🛒 Distance to supermarkets
- 🏗️ Build year / Property age
- 📐 Livable surface

These are among the most influential variables used by the XGBoost model.

Because their distributions have evolved, production data no longer perfectly matches the original training data.

---

## 📌 Notes About the Monitoring Report

Two observations deserve attention.

### 1. Redundant Features

`build_year` and `property_age` always produce the exact same PSI value (**0.9734**).

This is expected because:

```text
property_age = 2026 - build_year
```

Monitoring both variables is therefore redundant.

---

### 2. Price per m²

`price_per_m2` is still monitored even though it was intentionally removed from the model's training features because it introduced target leakage.

Its PSI therefore reflects differences between production logs and the training dataset rather than changes affecting the deployed model itself.

It may still be useful as an informational metric, but it should **not** be interpreted as evidence that a model input has drifted.

---

## 📌 Overall Conclusion

Despite the detected drift, the prediction service remains fully operational and continues to return valid price estimates.

However, the observed distribution shifts suggest that prediction quality is likely to deteriorate over time.

For this reason, the monitoring component recommends retraining the model using more recent real-estate data before the next production deployment.

# 🧪 Automated Testing

Automated tests verify that both the API and the prediction pipeline behave as expected.

---

## 📦 Install Test Dependencies

```bash
pip install pytest httpx
```

---

## ▶️ Run the Tests

Execute:

```bash
pytest tests/test_app.py
```

---

## ✅ Expected Output

```text
tests/test_app.py .... [100%]
================== 4 passed ==================
```

---

## ✔️ What Is Tested?

The automated test suite validates several critical aspects of the application:

- ❤️ API availability through the health endpoint.
- 🏠 Successful prediction using valid real estate data.
- 💰 Correct numeric prediction returned by the model.
- ❌ Validation of incorrect input types.
- ⚠️ Detection of missing required fields (HTTP 422).

Passing all tests confirms that the deployed application behaves as expected.

---

# 📚 Technologies Used

| Component | Technology |
|-----------|------------|
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

A trained **XGBoost** model was deployed behind a **FastAPI** REST API and connected to an interactive **Streamlit** web application, allowing users to estimate Belgian real estate prices in real time.

Beyond simply serving predictions, the project incorporates several software engineering and MLOps practices that improve reliability, maintainability, and production readiness.

## ✅ Implemented Features

- 🚀 REST API deployment with FastAPI
- 🎨 Interactive web interface with Streamlit
- 🧪 Automated testing using pytest
- ✔️ Input validation with Pydantic
- 📊 Logging of production predictions
- ❤️ Dedicated `/ping` health endpoint monitored by UptimeRobot
- 📉 Population Stability Index (PSI) monitoring
- 🔍 Automatic drift detection
- 🔄 Retraining recommendations based on monitoring results

---

## 📈 Key Findings

Re-running the evaluation and drift analysis highlighted two important observations.

### 🏰 Luxury Properties

The headline error metrics are largely driven by a handful of ultra-luxury properties (€5M+) that lie well outside the model's training distribution.

The model is therefore **not fundamentally inaccurate**; it is simply being asked to extrapolate far beyond the range of data it has learned from.

Segmenting the evaluation—or even training a dedicated luxury-property model—would provide a much more representative view of real-world performance.

---

### 📉 Drift Monitoring

The drift report currently monitors:

- two redundant variables (`build_year` and `property_age`), and
- one variable that no longer feeds the model (`price_per_m2`), which was removed because of target leakage.

While this does not affect the deployment itself, simplifying the monitored feature set would make the PSI report easier to interpret and more actionable.

---

## 🎯 Final Remarks

This project demonstrates that deploying a machine learning model is only one step in the production lifecycle.

A reliable ML system also requires:

- continuous monitoring,
- automated testing,
- robust data validation,
- regular retraining,
- and periodic verification that monitoring metrics remain aligned with the model's actual inputs.

Together, these practices help ensure that predictions remain reliable as production data evolves over time.

Overall, the project delivers a complete end-to-end deployment pipeline—from model serving and user interaction to monitoring, testing, and maintenance—following modern software engineering and MLOps best practices.

---

# 👤 Author

**Siegried Camus**

Developed as part of the **BeCode AI & Data Science Bootcamp**.
