import json
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
sys.path.append(str(BASE_DIR / "src"))

from src.features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features,
                          load_vocabulary, to_training_vocabulary)

THRESHOLD_PATH = MODEL_DIR / "luxury_threshold.json"


def load_routing_threshold(path: Path = THRESHOLD_PATH) -> float:
    """
    Routing threshold written by src/train.py next to the models it belongs to.

    No fallback: a guessed value silently changes which model prices a
    property, so a missing file or key stops the service at start-up instead.
    """
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["routing_probability_threshold"])


CALIBRATION_PATH = MODEL_DIR / "interval_calibration.json"


def load_interval_margin(path: Path = CALIBRATION_PATH) -> float:
    """
    Log-scale widening of the 10th-90th percentile range, from src/train.py's
    conformal calibration. No fallback: without it the raw quantile models
    hold the real price ~71% of the time while the UI promises 80%.
    """
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["log_margin"])


def route(result: dict, threshold: float) -> dict:
    """
    Pick the price the user gets from one engine result.

    At or above `threshold` the luxury model prices the property and no range
    is given: the range comes from the standard-segment quantile models and,
    next to a luxury price, would describe a different estimate. The API and
    the CLI both call this, so they cannot answer differently.
    """
    is_luxury = result["luxury_proba"] >= threshold
    return {
        "prediction": result["luxury_prediction"] if is_luxury else result["prediction"],
        "segment": "luxury" if is_luxury else "standard",
        "prediction_interval": None if is_luxury else result["prediction_interval"],
    }


class PredictEngine:
    """Loads the five model artifacts once and serves single-property predictions."""

    def __init__(self):
        self.pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
        self.pipeline_luxury = joblib.load(MODEL_DIR / "pipeline_luxury.joblib")
        self.luxury_classifier = joblib.load(MODEL_DIR / "luxury_classifier.joblib")
        self.pipeline_lower = joblib.load(MODEL_DIR / "pipeline_lower.joblib")
        self.pipeline_upper = joblib.load(MODEL_DIR / "pipeline_upper.joblib")
        self.interval_margin = load_interval_margin()
        self.vocabulary = load_vocabulary()

    def predict(self, data: dict):
        """
        Run every model on one property.

        Inputs are first rewritten into the training vocabulary (see
        to_training_vocabulary): without it the encoder ignored property type,
        province and state. Raises ValueError for categories the models never saw.
        """
        data, notes = to_training_vocabulary(data, self.vocabulary)
        df = pd.DataFrame([data])
        # A field sent as None becomes an object column; make it a numeric NaN
        # so add_features and the pipeline's median imputer treat it as missing.
        df = df.apply(lambda col: pd.to_numeric(col) if col.isna().all() else col)

        df = add_features(df)
        # The row the models read, JSON-safe (NaN -> None), for the drift log.
        features = json.loads(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
                              .to_json(orient="records"))[0]

        # Generate predictions
        standard_pred = float(np.expm1(self.pipeline.predict(df)[0]))
        luxury_pred = float(np.expm1(self.pipeline_luxury.predict(df)[0]))

        # Generate luxury probability
        luxury_proba = float(self.luxury_classifier.predict_proba(df)[0][1])

        # Generate prediction interval (quantile models)
        lower_pred = float(np.expm1(self.pipeline_lower.predict(df)[0] - self.interval_margin))
        upper_pred = float(np.expm1(self.pipeline_upper.predict(df)[0] + self.interval_margin))

        return {
            "prediction": standard_pred,
            "luxury_prediction": luxury_pred,
            "luxury_proba": luxury_proba,
            "prediction_interval": {
                "lower": lower_pred,
                "upper": upper_pred
            },
            "notes": notes,
            "features": features,
        }

engine = PredictEngine()