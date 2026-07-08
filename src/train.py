import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from features import add_features


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "clean" / "cleaned_data.json"
MODEL_DIR = BASE_DIR / "models"

XGB_MODEL_PATH = MODEL_DIR / "best_XGBoost.json"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

TARGET = "price"


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

CATEGORICAL_FEATURES = [
    "property_type",
    "province",
    "city",
    "property_state",
]


def load_data(path):
    df = pd.read_json(path)

    lower = df["price"].quantile(0.01)
    upper = df["price"].quantile(0.99)

    return df[(df["price"] >= lower) & (df["price"] <= upper)]


def clean_target(y):
    return y.clip(y.quantile(0.01), y.quantile(0.99))


def build_preprocessor():

    num = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=42))
    ])

    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", cat, CATEGORICAL_FEATURES)
    ])


def build_model():

    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        objective="reg:squarederror"
    )


def train():

    df = load_data(DATA_PATH)
    df = add_features(df)

    X = df.drop(columns=[TARGET])
    y = clean_target(np.log1p(df[TARGET]))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )

    preprocessor = build_preprocessor()

    X_train_p = preprocessor.fit_transform(X_train)

    model = build_model()

    model.fit(
        X_train_p,
        y_train
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save_model(
        XGB_MODEL_PATH
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print(f"Model saved at {XGB_MODEL_PATH}")
    print(f"Preprocessor saved at {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    train()