import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from features import add_features  # use the SAME feature engineering as train.py / evaluate.py

# Paths and evaluation
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
TEST_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"


def evaluate_model(json_path):
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found at: {PIPELINE_PATH}")

    # Load the combined preprocessor+model pipeline (saved via joblib.dump in train.py)
    pipeline = joblib.load(PIPELINE_PATH)
    df = pd.read_json(json_path)

    # 1. Extract target before feature engineering
    actual_prices = df["price"].values

    # 2. Add features
    df_processed = add_features(df.drop(columns=["price"]))

    # 3. Predict directly with the pipeline
    preds_log = pipeline.predict(df_processed)

    # 4. Inverse transform — training used np.log1p, so invert with np.expm1
    preds = np.expm1(preds_log)

    return preds, actual_prices


# Main execution
if __name__ == "__main__":
    if TEST_FILE.exists():
        predictions, actuals = evaluate_model(TEST_FILE)

        diffs = actuals - predictions
        abs_diffs = np.abs(diffs)

        # Metrics
        mae = np.mean(abs_diffs)
        rmse = np.sqrt(np.mean(diffs**2))
        mape = np.mean(abs_diffs / actuals) * 100
        mpe = np.mean(diffs / actuals) * 100

        print("\nPerformance evaluation")
        print(f"Total records evaluated : {len(predictions)}")
        print(f"MAE                     : €{mae:,.2f}")
        print(f"RMSE                    : €{rmse:,.2f}")
        print(f"MAPE                    : {mape:.2f}%")

        print(f"\nBias analysis")
        bias = "underestimating" if mpe > 0 else "overestimating"
        print(f"The model is {bias} prices by {abs(mpe):.2f}% on average.")

        # Outlier diagnostic helper
        print("\nTop 3 worst errors:")
        worst_indices = np.argsort(abs_diffs)[-3:]
        for idx in worst_indices:
            print(f"Actual: €{actuals[idx]:,.2f} | Predicted: €{predictions[idx]:,.2f} | Error: €{abs_diffs[idx]:,.2f}")
    else:
        print(f"Error: File not found at {TEST_FILE}")
