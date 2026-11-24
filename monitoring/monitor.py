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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILE = BASE_DIR / "logs.jsonl"
_LOCK = threading.Lock()

# PSI above this is strong drift (monitoring/check_drift.py reads it the same way).
STRONG_PSI = 0.25


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
        "timestamp": datetime.now(UTC).isoformat(),
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
    """
    Population Stability Index of `actual` against the reference `expected`.

    Bin edges are the deciles of the reference, so each bin holds about a tenth
    of it whatever the skew. Equal-width bins over a range like livable_surface
    (median ~150 m², max 5,000 m²) put 97% of the rows in the first bin, and a
    surface shift of x1.5 read as "OK". Values outside the reference range fall
    in the two end bins. A discrete feature yields fewer, merged edges.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    inner_edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    expected_hist = np.bincount(np.searchsorted(inner_edges, expected, side="right"),
                                minlength=len(inner_edges) + 1)
    actual_hist = np.bincount(np.searchsorted(inner_edges, actual, side="right"),
                              minlength=len(inner_edges) + 1)
    return share_psi(expected_hist / len(expected), actual_hist / len(actual))


def share_psi(expected_share, actual_share):
    """PSI between two aligned arrays of bin or category shares."""
    expected_pct = np.asarray(expected_share, dtype=float) + 1e-6
    actual_pct = np.asarray(actual_share, dtype=float) + 1e-6
    return float(np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct)))


def categorical_psi(expected, actual):
    """
    PSI on category shares, for the categorical inputs (province, city...).

    No binning: each category is its own bin, and a category seen only on one
    side counts too, so traffic that is all Brussels apartments shows up.
    """
    expected = pd.Series(expected).astype(str)
    actual = pd.Series(actual).astype(str)
    categories = sorted(set(expected) | set(actual))
    expected_share = expected.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual_share = actual.value_counts(normalize=True).reindex(categories, fill_value=0)
    return share_psi(expected_share.to_numpy(), actual_share.to_numpy())


def detect_drift(train_df: pd.DataFrame, live_df: pd.DataFrame, threshold=STRONG_PSI):
    """
    PSI of every column present in both frames, flagged above `threshold`.

    Numeric columns are binned on the training deciles (psi), the others are
    compared on category shares (categorical_psi). A column with no live value
    at all (say, a distance nobody fills in) gets psi None.
    """
    results = {}

    numerical_columns = set(train_df.select_dtypes(include=[np.number, bool]).columns)
    for col in train_df.columns:
        if col in live_df.columns:
            train_data = train_df[col].dropna()
            live_data = live_df[col].dropna()
            if live_data.empty:
                results[col] = {"psi": None, "drift_detected": False}
                continue
            if col in numerical_columns:
                score = psi(train_data.astype(float), pd.to_numeric(live_data, errors="coerce").dropna())
            else:
                score = categorical_psi(train_data, live_data)
            results[col] = {
                "psi": score,
                "drift_detected": score > threshold
            }
    return results
