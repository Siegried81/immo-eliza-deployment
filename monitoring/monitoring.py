import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

LOG_FILE = BASE_DIR / "logs.json"



# =====================================================
# PREDICTION LOGGER
# =====================================================

def log_prediction(input_data: dict, prediction: float):
    """
    Store API predictions for monitoring.
    """

    LOG_FILE.parent.mkdir(
        exist_ok=True
    )

    entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "input": input_data,

        "prediction": prediction
    }


    if LOG_FILE.exists():

        logs = json.loads(
            LOG_FILE.read_text(
                encoding="utf-8"
            )
        )

    else:

        logs = []


    logs.append(entry)


    LOG_FILE.write_text(
        json.dumps(
            logs,
            indent=2
        ),
        encoding="utf-8"
    )



# =====================================================
# PSI DRIFT METRIC
# =====================================================

def psi(
    expected,
    actual,
    bins=10
):
    """
    Population Stability Index.

    Higher PSI means stronger drift.

    < 0.1  : no significant drift
    0.1-0.25 : moderate drift
    > 0.25 : strong drift
    """


    expected = np.array(expected)
    actual = np.array(actual)


    expected_hist, bin_edges = np.histogram(
        expected,
        bins=bins
    )


    actual_hist, _ = np.histogram(
        actual,
        bins=bin_edges
    )


    expected_pct = (
        expected_hist /
        len(expected)
    )


    actual_pct = (
        actual_hist /
        len(actual)
    )


    psi_value = np.sum(
        (
            expected_pct - actual_pct
        )
        *
        np.log(
            (
                expected_pct + 1e-6
            )
            /
            (
                actual_pct + 1e-6
            )
        )
    )


    return float(psi_value)



# =====================================================
# DRIFT DETECTION
# =====================================================

def detect_drift(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    threshold=0.25
):
    """
    Compare training data distribution
    with live prediction data.

    Returns PSI score per numerical feature.
    """


    results = {}


    numerical_columns = (
        train_df
        .select_dtypes(
            include=np.number
        )
        .columns
    )


    for col in numerical_columns:

        if col in live_df.columns:

            score = psi(
                train_df[col].dropna(),
                live_df[col].dropna()
            )


            results[col] = {
                "psi": score,
                "drift_detected": score > threshold
            }


    return results