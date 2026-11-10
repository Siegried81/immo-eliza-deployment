"""
Evaluate the full routing pipeline (standard + luxury model) per price tier.

Two modes, on two different splits of src/train.py:
- default: sweeps the routing threshold on the VALIDATION split. That's where
  routing_probability_threshold is chosen, so these figures are optimistic.
- --test: scores the threshold in models/luxury_threshold.json (and no
  routing, for comparison) on the sealed TEST split. Nothing is chosen on it,
  so these are the figures to publish. Don't pick a threshold from them.

Tiers split at the luxury price threshold from models/luxury_threshold.json,
which is what the classifier was trained to detect.

Run:
    python scripts/evaluate_stratified.py           # choose the threshold
    python scripts/evaluate_stratified.py --test    # final figures
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))
from train import TARGET, load_dataset, price_tiers, split_all  # noqa: E402

MODEL_DIR = BASE_DIR / "models"
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]  # 1.01 = no routing


def main(on_test=False):
    _, val, _, test = split_all(load_dataset())
    rows_scored = test if on_test else val
    actual = rows_scored[TARGET].to_numpy()
    X_val = rows_scored.drop(columns=[TARGET])
    if on_test:
        print(f"Test split: {len(test)} rows (sealed: no model, search or threshold used them)\n")
    else:
        print(f"Validation split: {len(val)} rows (early stopping, tuning and this threshold "
              "were chosen on them, so the figures are optimistic)\n")

    config = json.loads((MODEL_DIR / "luxury_threshold.json").read_text())
    luxury_price = config["luxury_price_threshold"]
    thresholds = [config["routing_probability_threshold"], 1.01] if on_test else THRESHOLDS
    standard = np.expm1(joblib.load(MODEL_DIR / "pipeline.joblib").predict(X_val))
    luxury = np.expm1(joblib.load(MODEL_DIR / "pipeline_luxury.joblib").predict(X_val))
    proba = joblib.load(MODEL_DIR / "luxury_classifier.joblib").predict_proba(X_val)[:, 1]

    for threshold in thresholds:
        routed = proba >= threshold
        predicted = np.where(routed, luxury, standard)
        rows = []
        for name, low, high in price_tiers(luxury_price):
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
    main(on_test="--test" in sys.argv[1:])
