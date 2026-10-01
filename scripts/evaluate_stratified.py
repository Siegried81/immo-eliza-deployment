"""
Evaluate the full routing pipeline (standard + luxury model) per price tier,
for a range of routing thresholds, to choose routing_probability_threshold.

Scores train.py's validation split only. The earlier "holdout" (rows absent
from data/training_baseline.csv) still contained ~80% of the training rows,
so it rewarded models for memorising them. Tiers split at the luxury price
threshold from models/luxury_threshold.json, which is what the classifier
was trained to detect.

Run:
    python scripts/evaluate_stratified.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))
from features import add_features  # noqa: E402
from train import SPLIT_SEED, TARGET, TEST_SIZE  # noqa: E402

MODEL_DIR = BASE_DIR / "models"
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]  # 1.01 = no routing


def tiers(luxury_price: float) -> list[tuple[str, float, float]]:
    """Below €1M, €1M up to the luxury price, and the luxury segment."""
    return [("< €1M", 0, 1_000_000),
            (f"€1M - €{luxury_price / 1e6:.1f}M", 1_000_000, luxury_price),
            (f">= €{luxury_price / 1e6:.1f}M", luxury_price, float("inf"))]


def main():
    df = add_features(pd.read_json(BASE_DIR / "data" / "clean" / "cleaned_data.json"))
    _, val = train_test_split(df, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    actual = val[TARGET].to_numpy()
    X_val = val.drop(columns=[TARGET])
    print(f"Validation split: {len(val)} rows (never used to fit any model)\n")

    luxury_price = json.loads((MODEL_DIR / "luxury_threshold.json").read_text())["luxury_price_threshold"]
    standard = np.expm1(joblib.load(MODEL_DIR / "pipeline.joblib").predict(X_val))
    luxury = np.expm1(joblib.load(MODEL_DIR / "pipeline_luxury.joblib").predict(X_val))
    proba = joblib.load(MODEL_DIR / "luxury_classifier.joblib").predict_proba(X_val)[:, 1]

    for threshold in THRESHOLDS:
        routed = proba >= threshold
        predicted = np.where(routed, luxury, standard)
        rows = []
        for name, low, high in tiers(luxury_price):
            mask = (actual >= low) & (actual < high)
            err = predicted[mask] - actual[mask]
            rows.append({
                "Tier": name,
                "N": int(mask.sum()),
                "RoutedToLuxury": f"{routed[mask].sum()} ({routed[mask].mean():.1%})",
                "MAE": f"€{np.abs(err).mean():,.0f}",
                "MAPE": f"{np.mean(np.abs(err) / actual[mask]):.1%}",
                "Bias": f"{np.mean(err / actual[mask]):+.1%}",
            })
        overall = np.mean(np.abs(predicted - actual) / actual)
        label = "no routing" if threshold > 1 else f"THRESHOLD = {threshold}"
        print(f"{label}  (overall MAPE {overall:.1%}, routed {routed.sum()} rows)")
        print(pd.DataFrame(rows).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
