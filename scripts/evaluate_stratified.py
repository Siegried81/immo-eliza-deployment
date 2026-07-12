import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from features import add_features

# 1. Load data & artifacts
with open(BASE_DIR / "data" / "clean" / "cleaned_data.json") as f:
    cleaned = json.load(f)
cleaned_df = pd.DataFrame(cleaned)

baseline = pd.read_csv(BASE_DIR / "data" / "training_baseline.csv")
train_ids = set(baseline["property_id"])

# Filter to the TRUE holdout set
holdout_df = cleaned_df[~cleaned_df["property_id"].isin(train_ids)].copy()
print(f"Full evaluation set: {len(cleaned_df)} rows")
print(f"True holdout (excluding training leakage): {len(holdout_df)} rows\n")

# Feature engineering
holdout_df = add_features(holdout_df)

# Load full pipelines (preprocessor + model)
pipeline = joblib.load(BASE_DIR / "models" / "pipeline.joblib")
pipeline_luxury = joblib.load(BASE_DIR / "models" / "pipeline_luxury.joblib")
luxury_classifier = joblib.load(BASE_DIR / "models" / "luxury_classifier.joblib")

# 2. Precompute predictions
standard_points = []
luxury_points = []
luxury_probas = []

for _, row in holdout_df.iterrows():
    row_df = pd.DataFrame([row])
    # Apply expm1 to reverse log-transform
    standard_points.append(float(np.expm1(pipeline.predict(row_df)[0])))
    luxury_points.append(float(np.expm1(pipeline_luxury.predict(row_df)[0])))
    # Extract probability of luxury class (index 1)
    luxury_probas.append(float(luxury_classifier.predict_proba(row_df)[0][1]))

holdout_df["_standard_point"] = standard_points
holdout_df["_luxury_point"] = luxury_points
holdout_df["_luxury_proba"] = luxury_probas

# 3. Stratify and evaluate thresholds
def tier(price):
    if price < 1_000_000:
        return "< €1M"
    elif price < 3_000_000:
        return "€1M - €3M"
    else:
        return ">= €3M"

holdout_df["tier"] = holdout_df["price"].apply(tier)

# Finer grid, extended past 0.6 since MAPE on the <€1M tier was still
# improving monotonically at 0.6 in the last run.
THRESHOLDS = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for threshold in THRESHOLDS:
    routed_luxury = holdout_df["_luxury_proba"] >= threshold
    holdout_df["predicted_price"] = np.where(
        routed_luxury,
        holdout_df["_luxury_point"],
        holdout_df["_standard_point"]
    )

    rows = []
    for tier_name in ["< €1M", "€1M - €3M", ">= €3M"]:
        subset = holdout_df[holdout_df["tier"] == tier_name]
        if len(subset) == 0:
            continue

        actual = subset["price"].values
        predicted = subset["predicted_price"].values
        n_routed = int(routed_luxury[holdout_df["tier"] == tier_name].sum())

        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        bias = np.mean((predicted - actual) / actual) * 100

        rows.append({
            "Tier": tier_name,
            "N": len(subset),
            "RoutedToLuxury": f"{n_routed} ({100*n_routed/len(subset):.1f}%)",
            "MAE": f"€{mae:,.0f}",
            "MAPE": f"{mape:.1f}%",
            "Bias": f"{'+' if bias >= 0 else ''}{bias:.1f}%",
        })

    print(f"\nTHRESHOLD = {threshold}")
    print(pd.DataFrame(rows).to_string(index=False))

print("\nEvaluation complete. Select the threshold that balances MAPE/Bias on < €1M tier.")