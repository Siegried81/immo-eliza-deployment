"""
Write a synthetic prediction log, to check the drift report end to end.

The entries are validation rows the models never trained on, priced by the
standard model, so a correct report reads them as stable: this tests the
tooling, not production. They are tagged source="synthetic" and go to their
own file, never to the real prediction log.

Run:
    python -m monitoring.generate_logs [n]
    python -m monitoring.check_drift monitoring/synthetic_logs.jsonl
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from monitoring.monitor import log_prediction
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_LOG = PROJECT_ROOT / "monitoring" / "synthetic_logs.jsonl"
SEED = 42


def populate_logs(n=1500, path=SYNTHETIC_LOG):
    """Replace `path` with `n` validation rows (sampled with a fixed seed) and their predictions."""
    from src.train import TARGET, load_dataset, split

    _, val = split(load_dataset())
    sample = val.sample(n=n, replace=n > len(val), random_state=SEED)
    X = sample.drop(columns=[TARGET])
    predictions = np.expm1(joblib.load(PROJECT_ROOT / "models" / "pipeline.joblib").predict(X))
    features = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    path = Path(path)
    path.unlink(missing_ok=True)
    for (_, row), prediction in zip(features.iterrows(), predictions):
        row = row.where(row.notna(), None).to_dict()
        log_prediction({}, prediction, features=row, source="synthetic", path=path)
    print(f"Wrote {n} synthetic entries to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a synthetic prediction log.")
    parser.add_argument("n", nargs="?", type=int, default=1500, help="number of entries")
    parser.add_argument("--out", type=Path, default=SYNTHETIC_LOG, help="file to (re)write")
    args = parser.parse_args()
    populate_logs(args.n, args.out)
