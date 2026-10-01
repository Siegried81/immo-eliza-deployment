"""
Score the standard model (no luxury routing) on train.py's validation split,
overall and per price tier.

The validation split is the only data no model was fitted on (it does serve
early stopping for the standard model). The earlier "true holdout" (rows
absent from data/training_baseline.csv) still held ~80% of the training rows.
"""
import json
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from features import add_features
from train import SPLIT_SEED, TEST_SIZE

# Paths and evaluation
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
TEST_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"
LUXURY_PRICE = json.loads((BASE_DIR / "models" / "luxury_threshold.json").read_text())["luxury_price_threshold"]

# Price tiers used for stratified reporting, so a handful of luxury
# outliers can't dominate the headline metric. The top tier starts at the
# luxury price threshold, the segment the routing classifier detects.
PRICE_TIERS = [
    ("< €1M", 0, 1_000_000),
    (f"€1M - €{LUXURY_PRICE / 1e6:.1f}M", 1_000_000, LUXURY_PRICE),
    (f"€{LUXURY_PRICE / 1e6:.1f}M+", LUXURY_PRICE, float("inf")),
]


def evaluate_model(json_path):
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found at: {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)
    df = pd.read_json(json_path)
    _, df = train_test_split(df, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    print(f"Validation split: {len(df)} rows\n")

    actual_prices = df["price"].values
    df_processed = add_features(df.drop(columns=["price"]))

    preds_log = pipeline.predict(df_processed)
    preds = np.expm1(preds_log)                 # training used np.log1p, so invert with expm1

    return preds, actual_prices


def print_metrics(label, actuals, predictions):
    if len(actuals) == 0:
        print(f"{label:<15}: no records in this tier")
        return

    diffs = actuals - predictions
    abs_diffs = np.abs(diffs)
    mae = np.mean(abs_diffs)
    rmse = np.sqrt(np.mean(diffs**2))
    mape = np.mean(abs_diffs / actuals) * 100
    print(f"{label:<15}: n={len(actuals):<6} MAE=€{mae:>12,.2f}  RMSE=€{rmse:>12,.2f}  MAPE={mape:>6.2f}%")


if __name__ == "__main__":
    if TEST_FILE.exists():
        predictions, actuals = evaluate_model(TEST_FILE)

        diffs = actuals - predictions
        abs_diffs = np.abs(diffs)

        mae = np.mean(abs_diffs)
        rmse = np.sqrt(np.mean(diffs**2))
        mape = np.mean(abs_diffs / actuals) * 100
        mpe = np.mean(diffs / actuals) * 100

        print("Performance evaluation (validation split, standard model only)")
        print(f"Total records evaluated : {len(predictions)}")
        print(f"MAE                     : €{mae:,.2f}")
        print(f"RMSE                    : €{rmse:,.2f}")
        print(f"MAPE                    : {mape:.2f}%")

        print(f"\nBias analysis")
        bias = "underestimating" if mpe > 0 else "overestimating"
        print(f"The model is {bias} prices by {abs(mpe):.2f}% on average.")

        print("\nPerformance by price tier")
        for label, low, high in PRICE_TIERS:
            mask = (actuals >= low) & (actuals < high)
            print_metrics(label, actuals[mask], predictions[mask])

        print("\nTop 3 worst errors:")
        worst_indices = np.argsort(abs_diffs)[-3:]
        for idx in worst_indices:
            print(f"Actual: €{actuals[idx]:,.2f} | Predicted: €{predictions[idx]:,.2f} | Error: €{abs_diffs[idx]:,.2f}")
    else:
        print(f"Error: File not found at {TEST_FILE}")