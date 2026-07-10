import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
from features import add_features

PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
TEST_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"


def evaluate_model(json_path):
    """
        Loads a pre-trained machine learning pipeline and evaluates its performance 
        against a dataset of real estate properties.
        Args: json_path
        Returns: tuple: (predictions, actual_prices) where predictions are the model's 
                estimated prices and actual_prices are the ground truth values.
        Raises: FileNotFoundError: If the pipeline file is missing at the expected location.
        """
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found at: {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)   

    df = pd.read_json(json_path)
    actual_prices = df["price"].values

    df = add_features(df.drop(columns=["price"]))

    preds_log = pipeline.predict(df)
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