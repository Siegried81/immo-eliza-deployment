import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# Add src/ directly to sys.path
sys.path.insert(0, str(SRC_DIR))
from features import add_features

PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"


def get_user_input():
    data = {}
    print("\n--- IMMO ELIZA: PREDICTION CLI ---")

    data["postcode"] = int(input("Postcode: "))
    data["city"] = input("City: ").capitalize()
    data["province"] = input("Province: ")
    data["property_type"] = input("Type (HOUSE/APARTMENT): ").upper()
    data["property_state"] = input("State (e.g., NORMAL, NEW): ").upper()
    data["livable_surface"] = int(input("Livable surface (m²): "))
    data["build_year"] = int(input("Build year: ") or 2000)
    data["bedroom_count"] = int(input("Bedrooms: ") or 1)
    data["garage"] = int(input("Garages: ") or 0)
    data["garden_m2"] = int(input("Garden surface (m²): ") or 0)
    data["terrace"] = int(input("Terrace size (m²): ") or 0)
    data["swimming_pool"] = int(input("Pool (1/0): ") or 0)
    data["energy_consumption_kWh_m2_year"] = int(input("Energy consumption (kWh/m²/year): ") or 0)
    data["preschool_distance_m"] = int(input("Distance to preschool (m): ") or 0)
    data["train_station_distance_m"] = int(input("Distance to train station (m): ") or 0)
    data["supermarket_distance_m"] = int(input("Distance to supermarket (m): ") or 0)

    return data


def run_prediction():
    if not PIPELINE_PATH.exists():
        print(f"Error: Pipeline file not found in {PIPELINE_PATH.parent}")
        return

    pipeline = joblib.load(PIPELINE_PATH)

    # Prepare data
    df = pd.DataFrame([get_user_input()])
    df = add_features(df)

    # Predict
    pred_log = pipeline.predict(df)

    print(f"ESTIMATED PRICE: {np.expm1(pred_log)[0]:,.2f} €")


if __name__ == "__main__":
    run_prediction()