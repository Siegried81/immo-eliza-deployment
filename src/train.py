import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from xgboost import XGBRegressor
from category_encoders import TargetEncoder

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from features import add_features

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"
XGB_MODEL_PATH = MODEL_DIR / "best_XGBoost.json"
TRAINING_BASELINE_PATH = BASE_DIR / "data" / "training_baseline.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
PIPELINE_LOWER_PATH = MODEL_DIR / "pipeline_lower.joblib"
PIPELINE_UPPER_PATH = MODEL_DIR / "pipeline_upper.joblib"
PIPELINE_LUXURY_PATH = MODEL_DIR / "pipeline_luxury.joblib"
LUXURY_THRESHOLD_PATH = MODEL_DIR / "luxury_threshold.json"


"""
Define the target variable that the model will predict.
"""
TARGET = "price"


"""
List of numerical features used by the machine learning model.
'nearest_city_distance_km' has been removed to reduce drift.
'price_per_m2' has been removed: it was computed from the target (price),
which is data leakage.
"""
NUMERIC_FEATURES = [
    "build_year",
    "bedroom_count",
    "livable_surface",
    "total_surface",
    "garage",
    "terrace",
    "swimming_pool",
    "energy_consumption_kWh_m2_year",
    "property_state_encoded",
    "property_age",
    "preschool_distance_m",
    "train_station_distance_m",
    "supermarket_distance_m",
]


"""
List of categorical features used by the machine learning model.
"""
CATEGORICAL_FEATURES = [
    "property_type",
    "province",
    "city",
    "property_state",
]

QUANTILE_LOWER = 0.01
QUANTILE_UPPER = 0.99


def load_data(path):
    df = pd.read_json(path)
    lower = df["price"].quantile(QUANTILE_LOWER)
    upper = df["price"].quantile(QUANTILE_UPPER)

    return df[(df["price"] >= lower) & (df["price"] <= upper)]


"""
Limit extreme target values. Uses the same bounds as load_data() so the
target transform doesn't silently re-introduce a tighter cap.
"""
def clean_target(y):
    return y.clip(y.quantile(QUANTILE_LOWER), y.quantile(QUANTILE_UPPER))


"""
Create the preprocessing pipeline (still unfitted at this point).
"""
def build_preprocessor():
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder())
    ])

    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", cat, CATEGORICAL_FEATURES)
    ])


"""
Create the XGBoost regression model.
"""
def build_model(objective="reg:absoluteerror", quantile_alpha=None):
    params = dict(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        objective=objective,
        early_stopping_rounds=50
    )
    if quantile_alpha is not None:
        params["quantile_alpha"] = quantile_alpha
    return XGBRegressor(**params)


"""
Main training function.
"""
def train():
    # Load dataset and engineer features
    df = load_data(DATA_PATH)

    df = add_features(df)

    # Separate input features (X) from the target variable (y)
    X = df.drop(columns=[TARGET])
    y = clean_target(np.log1p(df[TARGET]))

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train, y_train)
    X_val_p = preprocessor.transform(X_val)

    # Train
    model = build_model()
    model.fit(
        X_train_p, y_train,
        eval_set=[(X_val_p, y_val)],
        verbose=False
    )

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(XGB_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    joblib.dump(pipeline, PIPELINE_PATH)

    print(f"Model saved at {XGB_MODEL_PATH}")
    print(f"Preprocessor saved at {PREPROCESSOR_PATH}")
    print(f"Combined pipeline saved at {PIPELINE_PATH}")


    lower_model = build_model(objective="reg:quantileerror", quantile_alpha=0.1)
    lower_model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)
    upper_model = build_model(objective="reg:quantileerror", quantile_alpha=0.9)
    upper_model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)

    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", lower_model)]), PIPELINE_LOWER_PATH)
    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", upper_model)]), PIPELINE_UPPER_PATH)
    print(f"Prediction interval models saved at {PIPELINE_LOWER_PATH} / {PIPELINE_UPPER_PATH}")
    print("These give a P10-P90 range around the point estimate; the app should treat any prediction")
    print("with a very wide P10-P90 gap as low-confidence.")

    full_df = add_features(pd.read_json(DATA_PATH))
    luxury_threshold = full_df["price"].quantile(QUANTILE_UPPER)
    luxury_df = full_df[full_df["price"] > luxury_threshold]

    with open(LUXURY_THRESHOLD_PATH, "w") as f:
        json.dump({"luxury_threshold": float(luxury_threshold), "luxury_model_available": len(luxury_df) >= 30}, f)

    if len(luxury_df) >= 30:
        X_lux = luxury_df.drop(columns=[TARGET])
        y_lux = np.log1p(luxury_df[TARGET])
        X_lux_train, X_lux_val, y_lux_train, y_lux_val = train_test_split(
            X_lux, y_lux, test_size=0.2, random_state=42
        )
        luxury_preprocessor = build_preprocessor()
        X_lux_train_p = luxury_preprocessor.fit_transform(X_lux_train, y_lux_train)
        X_lux_val_p = luxury_preprocessor.transform(X_lux_val)

        luxury_model = build_model()
        luxury_model.fit(X_lux_train_p, y_lux_train, eval_set=[(X_lux_val_p, y_lux_val)], verbose=False)

        luxury_pipeline = Pipeline([("preprocessor", luxury_preprocessor), ("model", luxury_model)])
        joblib.dump(luxury_pipeline, PIPELINE_LUXURY_PATH)
        print(f"Luxury-segment pipeline saved at {PIPELINE_LUXURY_PATH} "
              f"(trained on {len(luxury_df)} properties above €{luxury_threshold:,.0f})")
    else:
        print(f"Skipped luxury-segment model: only {len(luxury_df)} properties above €{luxury_threshold:,.0f} "
              f"(need at least 30). Gather more high-end listings before training this model.")

    # Save baseline for drift detection
    TRAINING_BASELINE_PATH.parent.mkdir(exist_ok=True)
    X_train.to_csv(TRAINING_BASELINE_PATH, index=False)
    print(f"Training baseline saved at {TRAINING_BASELINE_PATH}")

if __name__ == "__main__":
    train()
