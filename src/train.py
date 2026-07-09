import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from xgboost import XGBRegressor
from category_encoders import TargetEncoder

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from features import add_features

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"
XGB_MODEL_PATH = MODEL_DIR / "best_XGBoost.json"
TRAINING_BASELINE_PATH = BASE_DIR / "data" / "training_baseline.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"


"""
Define the target variable that the model will predict.
"""
TARGET = "price"


"""
List of numerical features used by the machine learning model.
'nearest_city_distance_km' has been removed to reduce drift.
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
    "price_per_m2",
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


"""
Load the cleaned dataset from a JSON file.
"""
def load_data(path):
    df = pd.read_json(path)
    lower = df["price"].quantile(0.01)
    upper = df["price"].quantile(0.99)
    return df[(df["price"] >= lower) & (df["price"] <= upper)]


"""
Limit extreme target values.
"""
def clean_target(y):
    return y.clip(y.quantile(0.01), y.quantile(0.99))


"""
Create the preprocessing pipeline.
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
def build_model():
    return XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        reg_lambda=1.5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        objective="reg:absoluteerror",
        early_stopping_rounds=50
    )


"""
Main training function.
"""
def train():
    # Load dataset and engineer features
    df = load_data(DATA_PATH)

    # Calculate price_per_m2 as a feature before add_features()
    df["price_per_m2"] = df["price"] / df["livable_surface"].replace(0, np.nan)
    df["price_per_m2"] = df["price_per_m2"].fillna(df["price_per_m2"].median())

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

    # Preprocess
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

    print(f"Model saved at {XGB_MODEL_PATH}")
    print(f"Preprocessor saved at {PREPROCESSOR_PATH}")

    # Save baseline for drift detection
    TRAINING_BASELINE_PATH.parent.mkdir(exist_ok=True)
    X_train.to_csv(TRAINING_BASELINE_PATH, index=False)
    print(f"Training baseline saved at {TRAINING_BASELINE_PATH}")

if __name__ == "__main__":
    train()
