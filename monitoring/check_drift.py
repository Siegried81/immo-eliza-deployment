"""
PSI drift report: the logged predictions against the rows the models were
trained on.

The baseline is train.py's training split, rebuilt from data/ with the shared
loader, so it is exactly what the models were fitted on. The live side is the
engineered features each log entry carries (monitoring/monitor.py), the same
columns the models read. The report says how many entries are real traffic
and how many are generated, because a PSI computed on generated data says
nothing about production.

Run:
    python -m monitoring.check_drift                      # the prediction log
    python -m monitoring.check_drift monitoring/synthetic_logs.jsonl
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from monitoring.monitor import STRONG_PSI, detect_drift, read_logs
from src.features import CATEGORICAL_FEATURES, MONITORED_FEATURES

# Below this many entries the PSI is mostly sampling noise. Drawing n rows from
# the validation split (no drift at all) 200 times, the noisiest feature's 95th
# percentile PSI is 0.22 at n=100, 0.12 at n=200 and 0.085 at n=300: from 300 on,
# a stable feature stays under the 0.1 "moderate" line.
MIN_SAMPLES = 300
MODERATE_PSI = 0.1


def drift_report(logs: list[dict], train_df: pd.DataFrame) -> tuple[dict, Counter]:
    """
    PSI per monitored feature, and the number of entries per source.

    Numeric features and the categorical ones (province, city, type, state) are
    both checked, for those the training frame has. Entries without engineered
    features (written before the log carried them) are skipped rather than
    guessed: their raw input lacks the encoded state.
    """
    usable = [entry for entry in logs if entry.get("features")]
    sources = Counter(entry.get("source", "unknown") for entry in usable)
    numeric = [f for f in MONITORED_FEATURES if f in train_df.columns]
    categorical = [f for f in CATEGORICAL_FEATURES if f in train_df.columns]
    live_df = pd.DataFrame([entry["features"] for entry in usable]).reindex(columns=numeric + categorical)
    live_df[numeric] = live_df[numeric].apply(pd.to_numeric, errors="coerce")
    return detect_drift(train_df[numeric + categorical], live_df), sources


def status(psi_score: float | None) -> str:
    """Usual PSI reading: below 0.1 stable, up to 0.25 moderate, above strong."""
    if psi_score is None:
        return "no live data"
    if psi_score > STRONG_PSI:
        return "STRONG DRIFT"
    return "Moderate" if psi_score > MODERATE_PSI else "OK"


def main(log_path: Path | None = None):
    from src.train import load_dataset, split

    logs = read_logs(log_path)
    usable = sum(1 for entry in logs if entry.get("features"))
    if usable < MIN_SAMPLES:
        print(f"Only {usable} usable log entries; need {MIN_SAMPLES}+ for a PSI worth reading.")
        return

    train_df, _ = split(load_dataset())
    results, sources = drift_report(logs, train_df)

    print(f"{usable} logged predictions vs {len(train_df)} training rows "
          f"(sources: {dict(sources)})")
    if set(sources) != {"api"}:
        print("Not only real traffic: this report does not describe production.")
    print(f"{'Feature':<32} | {'PSI':<8} | Status")
    print("-" * 60)
    for feature, result in results.items():
        shown = "-" if result["psi"] is None else f"{result['psi']:.4f}"
        print(f"{feature:<32} | {shown:<8} | {status(result['psi'])}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
