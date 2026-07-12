import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
sys.path.append(str(BASE_DIR / "src"))

from features import add_features

class PredictEngine:
    def __init__(self):
        self.pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
        self.pipeline_luxury = joblib.load(MODEL_DIR / "pipeline_luxury.joblib")
        self.luxury_classifier = joblib.load(MODEL_DIR / "luxury_classifier.joblib")

    def predict(self, data: dict):
        df = pd.DataFrame([data])

        df = add_features(df)
        # Generate predictions
        standard_pred = float(np.expm1(self.pipeline.predict(df)[0]))
        luxury_pred = float(np.expm1(self.pipeline_luxury.predict(df)[0]))
        # Generate luxury probability
        luxury_proba = float(self.luxury_classifier.predict_proba(df)[0][1])
        return {
            "prediction": standard_pred,
            "luxury_prediction": luxury_pred,
            "luxury_proba": luxury_proba
        }

engine = PredictEngine()