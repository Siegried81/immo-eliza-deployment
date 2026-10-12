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
    python monitoring/check_drift.py                      # the prediction log
    python monitoring/check_drift.py monitoring/synthetic_logs.jsonl
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from monitoring.monitor import detect_drift, read_logs  # noqa: E402
from src.features import MONITORED_FEATURES  # noqa: E402

MIN_SAMPLES = 50
MODERATE_PSI, STRONG_PSI = 0.1, 0.25


def drift_report(logs: list[dict], train_df: pd.DataFrame) -> tuple[dict, Counter]:
    """
    PSI per monitored feature, and the number of entries per source.

    Entries without engineered features (written before the log carried them)
    are skipped rather than guessed: their raw input lacks the encoded state.
    """
    usable = [entry for entry in logs if entry.get("features")]
    sources = Counter(entry.get("source", "unknown") for entry in usable)
    live_df = pd.DataFrame([entry["features"] for entry in usable])
    live_df = live_df.reindex(columns=MONITORED_FEATURES).apply(pd.to_numeric, errors="coerce")
    return detect_drift(train_df[MONITORED_FEATURES], live_df), sources


def status(psi_score: float | None) -> str:
    """Usual PSI reading: below 0.1 stable, up to 0.25 moderate, above strong."""
    if psi_score is None:
        return "no live data"
    if psi_score > STRONG_PSI:
        return "STRONG DRIFT"
    return "Moderate" if psi_score > MODERATE_PSI else "OK"


def main(log_path: Path | None = None):
    from train import load_dataset, split

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
