import pandas as pd
import numpy as np

def psi(expected, actual, bins=10):
    expected_hist, bins = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)

    psi_value = np.sum((expected_pct - actual_pct) * np.log((expected_pct + 1e-6) / (actual_pct + 1e-6)))
    return psi_value


def detect_drift(train_df, live_df):
    results = {}

    for col in train_df.select_dtypes(include=np.number).columns:
        if col in live_df.columns:
            results[col] = psi(train_df[col], live_df[col])

    return results