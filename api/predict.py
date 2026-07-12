import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
sys.path.append(str(BASE_DIR / "src"))

from src.features import add_features

class PredictEngine:
    def __init__(self):
        self.pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
        self.pipeline_luxury = joblib.load(MODEL_DIR / "pipeline_luxury.joblib")
        self.luxury_classifier = joblib.load(MODEL_DIR / "luxury_classifier.joblib")
        self.pipeline_lower = joblib.load(MODEL_DIR / "pipeline_lower.joblib")
        self.pipeline_upper = joblib.load(MODEL_DIR / "pipeline_upper.joblib")

    def predict(self, data: dict):
        df = pd.DataFrame([data])

        df = add_features(df)

        # Generate predictions
        standard_pred = float(np.expm1(self.pipeline.predict(df)[0]))
        luxury_pred = float(np.expm1(self.pipeline_luxury.predict(df)[0]))

        # Generate luxury probability
        luxury_proba = float(self.luxury_classifier.predict_proba(df)[0][1])

        # Generate prediction interval (quantile models)
        lower_pred = float(np.expm1(self.pipeline_lower.predict(df)[0]))
        upper_pred = float(np.expm1(self.pipeline_upper.predict(df)[0]))

        return {
            "prediction": standard_pred,
            "luxury_prediction": luxury_pred,
            "luxury_proba": luxury_proba,
            "prediction_interval": {
                "lower": lower_pred,
                "upper": upper_pred
            }
        }

engine = PredictEngine()