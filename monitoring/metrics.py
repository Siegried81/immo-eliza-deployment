import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from features import add_features 

# Paths and evaluation
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
TEST_FILE = BASE_DIR / "data" / "clean" / "cleaned_data.json"
TRAIN_BASELINE_PATH = BASE_DIR / "data" / "training_baseline.csv"

# Price tiers used for stratified reporting, so a handful of luxury
# outliers can't dominate the headline metric.
PRICE_TIERS = [
    ("< €1M", 0, 1_000_000),
    ("€1M - €3M", 1_000_000, 3_000_000),
    ("€3M+", 3_000_000, float("inf")),
]


def evaluate_model(json_path):
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found at: {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)
    df = pd.read_json(json_path)

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


def check_evaluation_overlap():
    
    if not TRAIN_BASELINE_PATH.exists():
        return
    try:
        train_df = pd.read_csv(TRAIN_BASELINE_PATH)
        test_df = pd.read_json(TEST_FILE)
        shared_cols = [c for c in ["livable_surface", "build_year", "bedroom_count"]
                       if c in train_df.columns and c in test_df.columns]
        if not shared_cols:
            return
        merged = test_df[shared_cols].merge(train_df[shared_cols].drop_duplicates(), on=shared_cols, how="inner")
        overlap_pct = 100 * len(merged) / max(len(test_df), 1)
        if overlap_pct > 5:
            print(f"\n⚠️  Provenance check: ~{overlap_pct:.1f}% of evaluation rows match training rows on "
                  f"build_year/bedroom_count/livable_surface. Worth confirming the evaluation set is a genuine "
                  f"hold-out and doesn't overlap with training data.")
    except Exception:
        pass 

if __name__ == "__main__":
    if TEST_FILE.exists():
        predictions, actuals = evaluate_model(TEST_FILE)

        diffs = actuals - predictions
        abs_diffs = np.abs(diffs)

        mae = np.mean(abs_diffs)
        rmse = np.sqrt(np.mean(diffs**2))
        mape = np.mean(abs_diffs / actuals) * 100
        mpe = np.mean(diffs / actuals) * 100

        print("\nPerformance evaluation (overall)")
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

        check_evaluation_overlap()
    else:
        print(f"Error: File not found at {TEST_FILE}")
