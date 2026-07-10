import sys
import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from monitoring.monitor import detect_drift

FEATURES_TO_MONITOR = [
    "build_year", "bedroom_count", "livable_surface", "total_surface",
    "garage", "terrace", "swimming_pool", "energy_consumption_kWh_m2_year",
    "property_state_encoded",
    "preschool_distance_m", "train_station_distance_m", "supermarket_distance_m"
]

def check_drift_report():
    logs_file = BASE_DIR / "monitoring" / "logs.json"
    baseline_file = BASE_DIR / "data" / "training_baseline.csv"

    if not logs_file.exists():
        print("Error: logs.json not found.")
        return

    with open(logs_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)

    if not logs or len(logs) < 50:
        print(f"Warning: Only {len(logs)} samples. Need 50+ for reliability.")
        return

    # Load data
    live_df = pd.DataFrame([log['input'] for log in logs])
    train_df = pd.read_csv(baseline_file)

    # Filter to common features
    live_df = live_df[FEATURES_TO_MONITOR]
    train_df = train_df[FEATURES_TO_MONITOR]

    print(f"Loaded {len(live_df)} live vs {len(train_df)} training samples.")
    print("Calculating PSI...")
    
    drift_results = detect_drift(train_df, live_df)

    print("DRIFT REPORT (PSI - Population Stability Index)")
    print(f"{'Feature':<30} | {'PSI':<10} | {'Status'}")
    print("-" * 70)

    total_drift = False
    for feature, result_data in drift_results.items():
        psi_score = result_data["psi"]
        if psi_score > 0.25:
            total_drift = True
            status = "🚨 STRONG DRIFT"
        elif psi_score > 0.1:
            status = "⚠️  Moderate"
        else:
            status = "✅ OK"
        print(f"{feature:<30} | {psi_score:<10.4f} | {status}")

    if total_drift:
        print("\nATTENTION: Strong drift detected! Retraining recommended.")
    else:
        print("\nNo significant drift - Model is valid.")

if __name__ == "__main__":
    check_drift_report()