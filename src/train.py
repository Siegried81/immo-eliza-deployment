import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from features import add_features

# Paths & Constants
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
PIPELINE_LUXURY_PATH = MODEL_DIR / "pipeline_luxury.joblib"
LUXURY_CLASSIFIER_PATH = MODEL_DIR / "luxury_classifier.joblib"
LUXURY_THRESHOLD_PATH = MODEL_DIR / "luxury_threshold.json"
PIPELINE_LOWER_PATH = MODEL_DIR / "pipeline_lower.joblib"
PIPELINE_UPPER_PATH = MODEL_DIR / "pipeline_upper.joblib"
TARGET = "price"

DEFAULT_ROUTING_THRESHOLD = 0.7

QUANTILE_LOWER = 0.10
QUANTILE_UPPER = 0.90


def build_preprocessor():
    """
    Constructs and returns a sklearn ColumnTransformer pipeline.
    Handles imputation and encoding for both numerical and categorical features.
    """
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", TargetEncoder())])
    return ColumnTransformer([
        ("num", num, ["build_year", "bedroom_count", "livable_surface", "total_surface", "garage", "terrace", "swimming_pool", "energy_consumption_kWh_m2_year", "property_state_encoded", "property_age", "preschool_distance_m", "train_station_distance_m", "supermarket_distance_m"]),
        ("cat", cat, ["property_type", "province", "city", "property_state"])
    ])


def train():
    """
    Executes the full training pipeline:
    1. Loads and engineers features.
    2. Trains standard regression and quantile interval models.
    3. Trains a luxury classification model and an optional luxury-specific regressor.
    4. Persists all models and configuration thresholds to disk.
    """
    df = add_features(pd.read_json(DATA_PATH))
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET].clip(df[TARGET].quantile(0.01), df[TARGET].quantile(0.99)))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train, y_train)
    X_val_p = preprocessor.transform(X_val)

    # Standard Regressor updated with Optuna parameters
    model = XGBRegressor(
        n_estimators=1468,
        learning_rate=0.027741436539627594,
        max_depth=9,
        reg_lambda=2.477519238111168,
        subsample=0.9261980084965962,
        colsample_bytree=0.5662663812979505,
        objective="reg:absoluteerror",
        random_state=42,
        early_stopping_rounds=50
    )
    model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", model)]), PIPELINE_PATH)

    # Prediction Interval (Quantile Regression) 
    lower_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE_LOWER,
        random_state=42,
        early_stopping_rounds=50
    )
    lower_model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)
    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", lower_model)]), PIPELINE_LOWER_PATH)

    upper_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE_UPPER,
        random_state=42,
        early_stopping_rounds=50
    )
    upper_model.fit(X_train_p, y_train, eval_set=[(X_val_p, y_val)], verbose=False)
    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", upper_model)]), PIPELINE_UPPER_PATH)

    # Luxury Routing Classifier 
    luxury_price_threshold = df[TARGET].quantile(0.99)
    y_lux = (df[TARGET] > luxury_price_threshold).astype(int)

    pos_weight = (y_lux == 0).sum() / max((y_lux == 1).sum(), 1)

    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=4,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42
    )

    clf.fit(X_train_p, y_lux.loc[X_train.index])

    # Train the luxury-segment regressor on luxury-labeled rows only
    X_lux_train = X_train[y_lux.loc[X_train.index] == 1]
    y_lux_train_price = y_train.loc[X_lux_train.index]
    if len(X_lux_train) >= 20:
        X_lux_train_p = preprocessor.transform(X_lux_train)
        luxury_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:absoluteerror",
            random_state=42
        )
        luxury_model.fit(X_lux_train_p, y_lux_train_price)
        joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", luxury_model)]), PIPELINE_LUXURY_PATH)
        luxury_model_available = True
    else:
        print(f"Warning: only {len(X_lux_train)} luxury rows in the training split — "
              f"skipping luxury regressor training (need >= 20).")
        luxury_model_available = False

    joblib.dump(Pipeline([("preprocessor", preprocessor), ("model", clf)]), LUXURY_CLASSIFIER_PATH)

    with open(LUXURY_THRESHOLD_PATH, "w") as f:
        json.dump({
            "luxury_price_threshold": float(luxury_price_threshold),
            "routing_probability_threshold": DEFAULT_ROUTING_THRESHOLD,
            "luxury_model_available": luxury_model_available
        }, f)

    print("Training complete. Models updated.")


if __name__ == "__main__":
    """
    Entry point for the training script.
    """
    train()