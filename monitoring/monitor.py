import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Define path to the log file inside the monitoring directory
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "logs.json"


def log_prediction(input_data: dict, prediction: float):
    """
    Store API predictions in a JSON file for model monitoring.
    Uses a robust approach to ensure data is appended correctly.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": float(prediction) # Ensure it's a standard float
    }

    # Use a try-except block to handle file access conflicts
    try:
        if LOG_FILE.exists():
            # Open with 'r+' mode to read, then we will overwrite
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        logs.append(entry)

        # Write the updated list back to the file
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        print(f"Error logging prediction: {e}")

def psi(expected, actual, bins=10):
    """
    Calculate the Population Stability Index (PSI) to measure distribution shift.
    """
    expected = np.array(expected)
    actual = np.array(actual)

    # Use combined bin edges for consistent comparison
    combined = np.concatenate([expected, actual])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # Add small constant to avoid division by zero
    expected_pct = (expected_hist / len(expected)) + 1e-6
    actual_pct = (actual_hist / len(actual)) + 1e-6

    psi_value = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
    return float(psi_value)


def detect_drift(train_df: pd.DataFrame, live_df: pd.DataFrame, threshold=0.25):
    """
    Compare training data distribution against live data
    and flag features exceeding the PSI threshold.
    """
    results = {}

    # Identify numerical columns in the training set
    numerical_columns = train_df.select_dtypes(include=np.number).columns

    # Iterate through columns and calculate drift if present in live data
    for col in numerical_columns:
        if col in live_df.columns:
            # Convert to float to handle bools/ints and prevent RuntimeWarnings
            train_data = train_df[col].dropna().astype(float)
            live_data = live_df[col].dropna().astype(float)
            
            # Calculate PSI score
            score = psi(
                train_data,
                live_data
            )

            # Store result and boolean flag indicating if drift is significant
            results[col] = {
                "psi": score,
                "drift_detected": score > threshold
            }

    return results