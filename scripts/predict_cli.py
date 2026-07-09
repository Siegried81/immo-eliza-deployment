import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

# This script lives at repo_root/scripts/predict_cli.py
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# Add src/ directly to sys.path (same convention as api/predict.py)
sys.path.insert(0, str(SRC_DIR))
from features import add_features

# Configuration
MODEL_PATH = BASE_DIR / "models" / "best_XGBoost.json"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

def get_user_input():
    data = {}
    print("\n--- IMMO ELIZA: PREDICTION CLI ---")

    data["postcode"] = int(input("Postcode: "))
    data["city"] = input("City: ").capitalize()
    data["province"] = input("Province: ")
    data["property_type"] = input("Type (HOUSE/APARTMENT): ").upper()
    data["property_state"] = input("State (e.g., NORMAL, NEW): ").upper()
    data["livable_surface"] = float(input("Livable surface (m²): "))
    data["build_year"] = int(input("Build year: ") or 2000)
    data["bedroom_count"] = int(input("Bedrooms: ") or 1)
    data["garage"] = int(input("Garages: ") or 0)
    data["garden_m2"] = float(input("Garden surface (m²): ") or 0)
    data["terrace"] = float(input("Terrace size (m²): ") or 0)
    data["swimming_pool"] = int(input("Pool (1/0): ") or 0)
    data["energy_consumption_kWh_m2_year"] = float(input("Energy consumption (kWh/m²/year): ") or 0)
    data["preschool_distance_m"] = float(input("Distance to preschool (m): ") or 0)
    data["train_station_distance_m"] = float(input("Distance to train station (m): ") or 0)
    data["supermarket_distance_m"] = float(input("Distance to supermarket (m): ") or 0)

    return data

def run_prediction():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        print(f"Error: Model files not found in {MODEL_PATH.parent}")
        return

    # Load resources
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Prepare data
    df = pd.DataFrame([get_user_input()])
    df = add_features(df)

    # Predict
    X = preprocessor.transform(df)
    pred_log = model.predict(X)

    print(f"ESTIMATED PRICE: {np.expm1(pred_log)[0]:,.2f} €")

if __name__ == "__main__":
    run_prediction()
