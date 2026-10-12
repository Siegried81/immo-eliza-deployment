"""
Compare the deployed hard routing threshold with probability blends of the
standard and luxury models, per price tier, on train.py's validation split.

A blend prices every property with both models, weighted by the luxury
probability in log space, so there is no jump at the threshold. The
classifier's probabilities are inflated by scale_pos_weight (~99), so the
script also tries them after isotonic calibration, fitted on one validation
half (calibration_halves) and judged on the other, which played no part in it.

Read-only: it loads the committed models and writes nothing.

Run:
    python scripts/evaluate_routing_blend.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))
from train import TARGET, calibration_halves, load_dataset, price_tiers, split  # noqa: E402

MODEL_DIR = BASE_DIR / "models"
DEPLOYED_THRESHOLD = 0.9


def blend(standard_log, luxury_log, proba):
    """Log-price mix of both models, the luxury share being the probability."""
    return proba * luxury_log + (1 - proba) * standard_log


def report(name, predicted, actual, luxury_price):
    """One line: overall MAPE, then MAPE and bias per price tier."""
    cells = []
    for tier, low, high in price_tiers(luxury_price):
        mask = (actual >= low) & (actual < high)
        err = predicted[mask] - actual[mask]
        cells.append(f"{tier} n={mask.sum()} MAPE={np.mean(np.abs(err) / actual[mask]):.1%} "
                     f"bias={np.mean(err / actual[mask]):+.1%}")
    overall = np.mean(np.abs(predicted - actual) / actual)
    print(f"{name:28} overall={overall:.1%} | " + " | ".join(cells))


def main():
    from sklearn.isotonic import IsotonicRegression

    _, val = split(load_dataset())
    actual = val[TARGET].to_numpy()
    X = val.drop(columns=[TARGET])
    standard = joblib.load(MODEL_DIR / "pipeline.joblib").predict(X)
    luxury = joblib.load(MODEL_DIR / "pipeline_luxury.joblib").predict(X)
    proba = joblib.load(MODEL_DIR / "luxury_classifier.joblib").predict_proba(X)[:, 1]
    luxury_price = json.loads((MODEL_DIR / "luxury_threshold.json").read_text())["luxury_price_threshold"]

    calib_index, _ = calibration_halves(val.index)
    calib = val.index.isin(calib_index)
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        proba[calib], (actual[calib] > luxury_price).astype(float))
    proba_cal = iso.predict(proba)

    for label, rows in [("All validation rows", np.ones(len(actual), bool)),
                        ("Check half (calibration never saw it)", ~calib)]:
        print(f"== {label}")
        candidates = {
            f"threshold {DEPLOYED_THRESHOLD} (deployed)": np.where(proba >= DEPLOYED_THRESHOLD, luxury, standard),
            "blend, raw probability": blend(standard, luxury, proba),
            "blend, isotonic probability": blend(standard, luxury, proba_cal),
            "threshold 0.5, isotonic": np.where(proba_cal >= 0.5, luxury, standard),
        }
        for name, predicted_log in candidates.items():
            report(name, np.expm1(predicted_log[rows]), actual[rows], luxury_price)
        print()


if __name__ == "__main__":
    main()
