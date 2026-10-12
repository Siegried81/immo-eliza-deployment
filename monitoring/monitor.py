"""
Prediction log and Population Stability Index (PSI) for drift monitoring.

The log is JSON Lines: one entry appended per prediction, so a request costs
the same whatever the log size, and a lock keeps two concurrent requests from
interleaving a line. Each entry carries the engineered features the models
read (from src/features.py), which is what drift has to be measured on.

The path defaults to monitoring/logs.jsonl. Set PREDICTION_LOG_PATH to a
persistent disk in production: Render's filesystem is wiped on every deploy
and restart, and a log that does not survive cannot be monitored.
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILE = BASE_DIR / "logs.jsonl"
_LOCK = threading.Lock()


def log_file() -> Path:
    """Where predictions are logged; read at call time so tests and deploys can move it."""
    return Path(os.getenv("PREDICTION_LOG_PATH") or DEFAULT_LOG_FILE)


def log_prediction(input_data: dict, prediction: float, features: dict | None = None,
                   source: str = "api", path: Path | None = None):
    """
    Append one prediction to the log.

    `source` tells real traffic from generated data, so a drift report can say
    which it measured. `path` overrides log_file(), for generated logs that
    must stay out of the real one. Logging never fails a prediction: an error
    is printed.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "input": input_data,
        "features": features or {},
        "prediction": float(prediction),
    }
    try:
        path = Path(path) if path else log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, default=str) + "\n"
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"Error logging prediction: {e}")


def read_logs(path: Path | None = None) -> list[dict]:
    """Every logged entry; a line cut short by a crash is skipped, not fatal."""
    path = Path(path) if path else log_file()
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


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


def detect_drift(train_df: pd.DataFrame, live_df: pd.DataFrame, threshold=0.25):
    """
    PSI of every numeric column present in both frames, flagged above `threshold`.

    A column with no live value at all (say, a distance nobody fills in) gets
    psi None rather than the NaN an empty histogram would produce.
    """
    results = {}

    numerical_columns = train_df.select_dtypes(include=[np.number, bool]).columns
    for col in numerical_columns:
        if col in live_df.columns:
            train_data = train_df[col].dropna().astype(float)
            live_data = live_df[col].dropna().astype(float)
            if live_data.empty:
                results[col] = {"psi": None, "drift_detected": False}
                continue
            score = psi(train_data, live_data)
            results[col] = {
                "psi": score,
                "drift_detected": score > threshold
            }
    return results
