import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from features import add_features

MODEL_PATH = BASE_DIR / "models" / "best_XGBoost.joblib"


class PredictionEngine:

    def __init__(self):

        package = joblib.load(MODEL_PATH)

        self.model = package["model"]
        self.preprocessor = package["preprocessor"]

    def predict(self, data: dict):

        df = pd.DataFrame([data])

        df = add_features(df)

        X = self.preprocessor.transform(df)

        prediction = self.model.predict(X)

        return float(np.expm1(prediction)[0])


engine = PredictionEngine()