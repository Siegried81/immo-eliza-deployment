import json
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
sys.path.append(str(BASE_DIR / "src"))

from src.features import add_features, load_vocabulary, to_training_vocabulary

class PredictEngine:
    """Loads the five model artifacts once and serves single-property predictions."""

    def __init__(self):
        self.pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
        self.pipeline_luxury = joblib.load(MODEL_DIR / "pipeline_luxury.joblib")
        self.luxury_classifier = joblib.load(MODEL_DIR / "luxury_classifier.joblib")
        self.pipeline_lower = joblib.load(MODEL_DIR / "pipeline_lower.joblib")
        self.pipeline_upper = joblib.load(MODEL_DIR / "pipeline_upper.joblib")
        # Conformal widening of the quantile range (see src/train.py::conformal_margin);
        # without it the raw 10th-90th percentile models cover ~71% instead of 80%.
        calibration = MODEL_DIR / "interval_calibration.json"
        self.interval_margin = json.loads(calibration.read_text())["log_margin"] if calibration.exists() else 0.0
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
        }

engine = PredictEngine()