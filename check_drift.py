import pandas as pd
import json
from pathlib import Path
from monitoring.monitor import detect_drift

def check_drift_report():
    """
    Analyze drift between training data and live predictions.
    """
    logs_file = Path("monitoring/logs.json")
    baseline_file = Path("data/training_baseline.csv")

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
        # Extraction de la valeur PSI depuis le dictionnaire
        psi_score = result_data["psi"]
        
        if psi_score > 0.25:
            total_drift = True
            status = "🚨 STRONG DRIFT"
        elif psi_score > 0.1:
            status = "⚠️  Moderate"
        else:
            status = "✅ OK"
        
        print(f"{feature:<25} | {psi_score:<10.4f} | {status}")

    print("=" * 70)
    if total_drift:
        print("ATTENTION: Strong drift detected! Retraining recommended.")
    else:
        print("No significant drift - Model is valid.")
    print("=" * 70)

if __name__ == "__main__":
    check_drift_report()