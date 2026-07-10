import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "logs.json"

def log_prediction(input_data: dict, prediction: float):
    """Store API predictions in a JSON file for model monitoring."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": float(prediction)
    }
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                try: logs = json.load(f)
                except json.JSONDecodeError: logs = []
        else: logs = []
        logs.append(entry)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error logging prediction: {e}")

def psi(expected, actual, bins=10):
    """Calculate the Population Stability Index (PSI)."""
    expected = np.array(expected)
    actual = np.array(actual)
    combined = np.concatenate([expected, actual])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)
    expected_pct = (expected_hist / len(expected)) + 1e-6
    actual_pct = (actual_hist / len(actual)) + 1e-6
    psi_value = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
    return float(psi_value)

def calculate_psi(expected_df: pd.DataFrame, actual_df: pd.DataFrame):
    """Wrapper to calculate PSI for all numerical columns."""
    psi_values = {}
    for col in expected_df.columns:
        if col in actual_df.columns and pd.api.types.is_numeric_dtype(expected_df[col]):
            psi_values[col] = psi(expected_df[col].dropna(), actual_df[col].dropna())
    return psi_values

def detect_drift(train_df: pd.DataFrame, live_df: pd.DataFrame, threshold=0.25):
    """Compare training data distribution against live data."""
    results = {}
    numerical_columns = train_df.select_dtypes(include=np.number).columns
    for col in numerical_columns:
        if col in live_df.columns:
            train_data = train_df[col].dropna().astype(float)
            live_data = live_df[col].dropna().astype(float)
            score = psi(train_data, live_data)
            results[col] = {
                "psi": score,
                "drift_detected": score > threshold
            }
    return results
