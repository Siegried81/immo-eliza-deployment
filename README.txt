Voici un README complet adapté à ton projet **Immo Eliza Deployment**, avec ton architecture actuelle (**FastAPI + Docker + Streamlit + modèle XGBoost**) et les instructions demandées. Tu peux le mettre directement dans `README.md`.

```markdown
# 🏠 Immo Eliza Deployment

## Mission

Immo Eliza is a Belgian real estate company that wants to make its machine learning price prediction model accessible to both developers and non-technical users.

This project deploys a trained regression model through:

- A **FastAPI backend** providing property price predictions through a REST API.
- A **Streamlit frontend application** allowing users to interact with the prediction system.
- A machine learning model trained with **XGBoost** to estimate Belgian property prices.

The API and the web application are connected but remain independent, following a clean deployment architecture.

---

# 🏗️ Architecture

The project follows the combined deployment approach:

```

```
                User
                 |
                 |
          Streamlit Application
                 |
                 |
          HTTP POST Request
                 |
                 |
          FastAPI Prediction API
                 |
                 |
          Prediction Engine
                 |
                 |
    XGBoost Model + Preprocessing Pipeline
```

```


## Components

### Backend API

The FastAPI application:

- Receives property information in JSON format.
- Applies the same preprocessing used during training.
- Loads the trained XGBoost model.
- Returns the predicted property price.

### Frontend Application

The Streamlit application:

- Provides an easy interface for non-technical users.
- Collects property characteristics.
- Sends requests to the API.
- Displays the predicted price.

---

# 📁 Project Structure

```

immo-eliza-deployment
│
├── api
│   │
│   ├── app.py                  # FastAPI application
│   ├── predict.py              # Model loading and prediction logic
│   ├── Dockerfile              # Docker configuration
│   │
│   ├── models
│   │   └── best_XGBoost.joblib # Trained model artifacts
│   │
│   └── requirements.txt
│
│
├── monitoring
│   │
│   ├── monitoring.py            # Prediction logging and drift detection
│   └── logs.json
│
│
├── src
│   │
│   └── features.py              # Feature engineering pipeline
│
│
├── streamlit
│   │
│   └── app.py                   # Streamlit frontend application
│
│
├── README.md
├── requirements.txt
└── .gitignore

```

---

# 🤖 Machine Learning Model

The prediction model was trained in the previous Immo Eliza machine learning project.

The final selected model is:

**XGBoost Regressor**

Reasons for choosing XGBoost:

- Strong performance on structured tabular data.
- Handles nonlinear relationships between property characteristics and prices.
- Good balance between accuracy and inference speed.

The model artifact contains:

- The trained XGBoost model.
- The preprocessing pipeline.
- Feature transformations required before prediction.

The model predicts:

```

Property features → Preprocessing → XGBoost → Estimated price (€)

````

---

# 🔌 API

## Technologies

- FastAPI
- Pydantic
- Scikit-learn preprocessing pipeline
- XGBoost
- Docker

---

# API Routes

## Health Check

### GET `/`

Checks whether the API is running.

### Response

```json
{
  "status": "API running"
}
````

---

## Property Prediction

### POST `/predict`

Receives property information and returns a predicted price.

---

## Input Example

```json
{
  "postcode": 1000,
  "province": "Brussels",
  "city": "Brussels",

  "property_type": "APARTMENT",
  "property_state": "NORMAL",

  "livable_surface": 80,
  "total_surface": 100,
  "bedroom_count": 2,
  "build_year": 2000,

  "garage": 0,
  "garden_m2": 0,
  "terrace": 10,
  "swimming_pool": false,

  "energy_consumption_kWh_m2_year": 130,

  "preschool_distance_m": 500,
  "train_station_distance_m": 800,
  "supermarket_distance_m": 400,

  "nearest_city": "Brussels",
  "nearest_city_distance_km": 0
}
```

---

## Output Example

```json
{
  "prediction": 285000.50,
  "currency": "EUR",
  "status": "success"
}
```

---

# Prediction Pipeline

The API prediction process:

1. Receive JSON input.
2. Convert JSON into a pandas DataFrame.
3. Apply feature engineering.
4. Apply preprocessing pipeline.
5. Generate prediction using XGBoost.
6. Convert prediction back from logarithmic scale.
7. Return JSON response.

---

# 🐳 Docker Deployment

The API is containerized using Docker.

The Docker image contains:

* Python environment.
* Required dependencies.
* FastAPI application.
* Trained model artifacts.

Build the Docker image:

```bash
docker build -t immo-eliza-api .
```

Run locally:

```bash
docker run -p 8000:8000 immo-eliza-api
```

The API will be available at:

```
http://localhost:8000
```

FastAPI documentation:

```
http://localhost:8000/docs
```

---

# 🌐 Streamlit Application

The Streamlit application provides a user-friendly interface.

Users can enter:

### Location

* Postal code
* Province
* City

### Property information

* Property type
* Building state
* Living surface
* Total surface
* Bedrooms
* Construction year
* Garage
* Garden
* Terrace
* Swimming pool

### Energy and distances

* Energy consumption
* Distance to schools
* Distance to train stations
* Distance to supermarkets

The application sends the information to the FastAPI backend and displays the estimated property price.

---

# 📦 Installation

## Clone repository

```bash
git clone https://github.com/<username>/immo-eliza-deployment.git

cd immo-eliza-deployment
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

## Start API

From the project root:

```bash
uvicorn api.app:app --reload
```

API:

```
http://127.0.0.1:8000
```

---

## Start Streamlit

In another terminal:

```bash
streamlit run streamlit/app.py
```

Application:

```
http://localhost:8501
```

---

# 📊 Monitoring

The project includes a monitoring system.

It provides:

## Prediction logging

Every API prediction can be stored with:

* Timestamp
* Input features
* Prediction result

Example:

```json
{
  "timestamp": "2026-07-07T10:00:00",
  "input": {
    "property_type": "APARTMENT"
  },
  "prediction": 285000
}
```

---

## Data Drift Detection

The project includes a Population Stability Index (PSI) implementation.

It compares:

* Training data distribution
* New prediction data distribution

Interpretation:

| PSI        | Meaning              |
| ---------- | -------------------- |
| < 0.1      | No significant drift |
| 0.1 - 0.25 | Moderate drift       |
| > 0.25     | Strong drift         |

---

# 🚀 Deployment

## Backend

Deployment target:

* Docker container
* Render

Required files:

```
api/
├── app.py
├── predict.py
├── Dockerfile
└── requirements.txt
```

---

## Frontend

Deployment target:

* Streamlit Community Cloud

Required files:

```
streamlit/
└── app.py
```

---

# ✅ Deliverables Checklist

| Requirement                                   | Status |
| --------------------------------------------- | ------ |
| Machine learning model trained and predicting | ✅      |
| FastAPI API created                           | ✅      |
| Prediction endpoint available                 | ✅      |
| Dockerfile created                            | ✅      |
| Streamlit application created                 | ✅      |
| API and frontend connected                    | ✅      |
| Model artifacts included                      | ✅      |
| Clean GitHub repository                       | ✅      |

---

# 👩‍💻 Author

Immo Eliza Deployment project developed as part of the BeCode Machine Learning training.

```

Ce README correspond à ton architecture actuelle et couvre les critères d'évaluation :
- modèle entraîné ✅
- API FastAPI ✅
- Dockerfile prévu ✅
- Streamlit ✅
- monitoring mentionné ✅
- structure GitHub propre ✅
```
