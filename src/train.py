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
These features are processed with numerical preprocessing steps.
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
These features require encoding before training.
"""
CATEGORICAL_FEATURES = [
    "property_type",
    "province",
    "city",
    "property_state",
]



"""
Load the cleaned dataset from a JSON file.

The function removes extreme price values by keeping
only properties between the 1st and 99th price percentiles.
This reduces the impact of outliers during training.
"""
def load_data(path):

    df = pd.read_json(path)

    lower = df["price"].quantile(0.01)

    upper = df["price"].quantile(0.99)

    return df[
        (df["price"] >= lower) &
        (df["price"] <= upper)
    ]



"""
Limit extreme target values.

This prevents very high or very low prices from having
too much influence on model training.
"""
def clean_target(y):

    return y.clip(
        y.quantile(0.01),
        y.quantile(0.99)
    )



"""
Create the preprocessing pipeline.

Numerical features:
- Missing values are completed using IterativeImputer.

Categorical features:
- Missing values are replaced with the most frequent value.
- Categories are converted into numerical values using OneHotEncoder.
"""
def build_preprocessor():

    num = Pipeline([
        (
            "imputer",
            IterativeImputer(
                max_iter=10,
                random_state=42
            )
        )
    ])


    cat = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])


    return ColumnTransformer([
        (
            "num",
            num,
            NUMERIC_FEATURES
        ),
        (
            "cat",
            cat,
            CATEGORICAL_FEATURES
        )
    ])



"""
Create the XGBoost regression model.

The hyperparameters are configured to balance
prediction performance and model generalization.
"""
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



"""
Main training function.

Workflow:
1. Load and clean data.
2. Generate additional features.
3. Split data into training, validation, and test sets.
4. Apply preprocessing.
5. Train the XGBoost model.
6. Save the trained model and preprocessor.
"""
def train():


    """
    Load the dataset and create additional engineered features.
    """
    df = load_data(DATA_PATH)

    df = add_features(df)



    """
    Separate input features (X) from the target variable (y).

    The target price is transformed using log1p
    to reduce the effect of price distribution skewness.
    """
    X = df.drop(
        columns=[TARGET]
    )

    y = clean_target(
        np.log1p(df[TARGET])
    )



    """
    Split the dataset into training and temporary datasets.

    70% is used for training.
    30% is reserved for validation and testing.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )


    """
    Split the remaining data equally.

    15% validation set.
    15% test set.
    """
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )



    """
    Create the preprocessing pipeline and transform training data.
    """
    preprocessor = build_preprocessor()

    X_train_p = preprocessor.fit_transform(
        X_train
    )



    """
    Create and train the XGBoost regression model.
    """
    model = build_model()

    model.fit(
        X_train_p,
        y_train
    )



    """
    Create the model directory if it does not exist.
    """
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )



    """
    Save the trained XGBoost model.
    """
    model.save_model(
        XGB_MODEL_PATH
    )


    """
    Save the preprocessing pipeline.
    It is required later to transform new prediction data.
    """
    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )



    """
    Display the location of saved files.
    """
    print(
        f"Model saved at {XGB_MODEL_PATH}"
    )

    print(
        f"Preprocessor saved at {PREPROCESSOR_PATH}"
    )
    
    """
    Save baseline for drift detection.
    """
    TRAINING_BASELINE_PATH.parent.mkdir(exist_ok=True)
    X_train.to_csv(TRAINING_BASELINE_PATH, index=False)

    print(f"Training baseline saved at {TRAINING_BASELINE_PATH}")

if __name__ == "__main__":

    train()