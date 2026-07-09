import sys
import pandas as pd
import json
from pathlib import Path

# This file lives at repo_root/monitoring/check_drift.py
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Package-style import (consistent with generate_logs.py) instead of the
# previous "from monitor import detect_drift", which only worked if this
# script happened to be run with cwd = monitoring/
from monitoring.monitor import detect_drift

def check_drift_report():
    """
    Analyze drift between training data and live predictions.
    """
    # Paths resolved from BASE_DIR, not from the current working directory,
    # so this script works no matter where it's invoked from.
    logs_file = BASE_DIR / "monitoring" / "logs.json"
    baseline_file = BASE_DIR / "data" / "training_baseline.csv"

    # 1. Load logged predictions
    if not logs_file.exists():
        print(f"Error: {logs_file} not found.")
        return

    with open(logs_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)

    if not logs:
        print("Error: Logs file is empty.")
        return

    # Extract the 'input' dictionary from each log entry
    live_df = pd.DataFrame([log['input'] for log in logs])
    print(f"Successfully loaded {len(live_df)} live predictions.")

    # 2. Load training baseline
    if not baseline_file.exists():
        print(f"Error: {baseline_file} not found.")
        return

    train_df = pd.read_csv(baseline_file)
    print(f"Successfully loaded {len(train_df)} training samples.")

    # 3. Detect drift
    print("Calculating PSI values...")
    drift_results = detect_drift(train_df, live_df)

    # 4. Report results
    if not drift_results:
        print("No numerical features found to compare.")
        return

    print("\n" + "="*70)
    print("DRIFT REPORT (PSI - Population Stability Index)")
    print("="*70)
    print(f"{'Feature':<25} | {'PSI':<10} | {'Status'}")
    print("-" * 70)

    total_drift = False
    for feature, result_data in drift_results.items():
        # Extract the PSI value from the result dict
        psi_score = result_data["psi"]

        if psi_score > 0.25:
            total_drift = True
            status = "🚨 STRONG DRIFT"
        elif psi_score > 0.1:
            status = "⚠️  Moderate"
        else:
            status = "✅ OK"

        print(f"{feature:<25} | {psi_score:<10.4f} | {status}")

    if total_drift:
        print("ATTENTION: Strong drift detected! Retraining recommended.")
    else:
        print("No significant drift - Model is valid.")

if __name__ == "__main__":
    check_drift_report()
