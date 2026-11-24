"""
Score the standard model (no luxury routing) on train.py's sealed test split,
overall and per price tier.

The test split is the only data nothing was fitted, early-stopped, tuned or
chosen on. The validation split did all of that, so it's no place for a
headline figure. A holdout made of rows "absent from the training set" is not
enough either: it can still contain most of the training rows.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import r2_score

from src.train import DATA_PATH, TARGET, load_dataset, price_tiers, split_all

BASE_DIR = Path(__file__).resolve().parent.parent

PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
LUXURY_PRICE = json.loads((BASE_DIR / "models" / "luxury_threshold.json").read_text())["luxury_price_threshold"]

# Price tiers used for stratified reporting, so a handful of luxury
# outliers can't dominate the headline metric. The top tier starts at the
# luxury price threshold, the segment the routing classifier detects.
PRICE_TIERS = price_tiers(LUXURY_PRICE)


def evaluate_model():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found at: {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)
    _, _, _, df = split_all(load_dataset())
    print(f"Test split: {len(df)} rows\n")

    actual_prices = df[TARGET].values
    df_processed = df.drop(columns=[TARGET])

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
    if DATA_PATH.exists():
        predictions, actuals = evaluate_model()

        diffs = actuals - predictions
        abs_diffs = np.abs(diffs)

        mae = np.mean(abs_diffs)
        rmse = np.sqrt(np.mean(diffs**2))
        mape = np.mean(abs_diffs / actuals) * 100
        mpe = np.mean(diffs / actuals) * 100

        print("Performance evaluation (test split, standard model only)")
        print(f"Total records evaluated : {len(predictions)}")
        print(f"MAE                     : €{mae:,.2f}")
        print(f"RMSE                    : €{rmse:,.2f}")
        print(f"MAPE                    : {mape:.2f}%")
        print(f"R² (euros)              : {r2_score(actuals, predictions):.4f}")

        print("\nBias analysis")
        bias = "underestimating" if mpe > 0 else "overestimating"
        print(f"The model is {bias} prices by {abs(mpe):.2f}% on average.")
        # The mean above is pulled up by the cheap listings (an estimate at
        # twice the price weighs +100%, one at half the price -50%), so the
        # median says which way a typical estimate leans.
        print(f"Median signed error     : {np.median(-diffs / actuals) * 100:+.2f}% (above 0: too high)")
        print(f"Median APE              : {np.median(abs_diffs / actuals) * 100:.2f}%")

        print("\nPerformance by price tier")
        for label, low, high in PRICE_TIERS:
            mask = (actuals >= low) & (actuals < high)
            print_metrics(label, actuals[mask], predictions[mask])

        print("\nTop 3 worst errors:")
        worst_indices = np.argsort(abs_diffs)[-3:]
        for idx in worst_indices:
            print(f"Actual: €{actuals[idx]:,.2f} | Predicted: €{predictions[idx]:,.2f} | Error: €{abs_diffs[idx]:,.2f}")
    else:
        print(f"Error: File not found at {DATA_PATH}")
