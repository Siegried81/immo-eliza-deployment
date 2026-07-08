import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from features import add_features


MODEL_PATH = BASE_DIR / "models" / "best_XGBoost.json"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"


class PredictionEngine:

    def __init__(self):

        self.model = xgb.XGBRegressor()

        self.model.load_model(
            MODEL_PATH
        )

        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )


    def predict(self, data: dict):

        df = pd.DataFrame([data])

        df = add_features(df)

        X = self.preprocessor.transform(df)

        prediction = self.model.predict(X)

        return float(
            np.expm1(prediction)[0]
        )


engine = PredictionEngine()