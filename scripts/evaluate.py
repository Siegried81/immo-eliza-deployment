import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

# This script lives at repo_root/scripts/evaluate.py
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

# Add src/ directly to sys.path so "features" imports the same way
# api/predict.py does, instead of requiring "from src.features import ..."
sys.path.insert(0, str(SRC_DIR))
from features import add_features

# Paths
MODEL_PATH = BASE_DIR / "models" / "best_XGBoost.json"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
TEST_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"

def evaluate_model(json_path):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    # Load model and preprocessor
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    df = pd.read_json(json_path)
    actual_prices = df["price"].values

    # Process features
    df = add_features(df.drop(columns=["price"]))

    # Predict
    X = preprocessor.transform(df)
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)

    return preds, actual_prices

if __name__ == "__main__":
    if TEST_FILE.exists():
        predictions, actuals = evaluate_model(TEST_FILE)

        diffs = actuals - predictions
        abs_diffs = np.abs(diffs)

        mae = np.mean(abs_diffs)
        rmse = np.sqrt(np.mean(diffs**2))
        mape = np.mean(abs_diffs / actuals) * 100

        print("\nPerformance Evaluation")
        print(f"Total Records Evaluated : {len(predictions)}")
        print(f"MAE                     : €{mae:,.2f}")
        print(f"RMSE                    : €{rmse:,.2f}")
        print(f"MAPE                    : {mape:.2f}%")
    else:
        print(f"Error: Dataset not found at {TEST_FILE}")
