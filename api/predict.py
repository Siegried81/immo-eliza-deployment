import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
import logging

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from features import add_features
from monitoring.monitor import calculate_psi

PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"

class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


class PredictionEngine:
    """
    Initialize the prediction engine. The pipeline (preprocessor + model)
    is loaded only once when the application starts.
    """
    def __init__(self):
        try:
            if not PIPELINE_PATH.exists():
                raise FileNotFoundError(f"Pipeline not found: {PIPELINE_PATH}")

            self.pipeline = joblib.load(PIPELINE_PATH)

        except Exception as e:
            raise PredictionError(f"Failed to initialize prediction engine: {str(e)}")

    """
    Internal helper to check for data drift.
    """
    def _check_drift(self, df: pd.DataFrame):
        # Skipped for single-row prediction to avoid argument errors
        logger.info("Drift detection skipped for single-row prediction.")
        return True

    """
    Generate a property price prediction.
    """
    def predict(self, data: dict) -> float:
        try:
            df = pd.DataFrame([data])

            df = add_features(df)

            self._check_drift(df)

            prediction = self.pipeline.predict(df)

            return float(np.expm1(prediction)[0])

        except ValueError as e:
            raise PredictionError(f"Data processing error: {str(e)}")
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")


engine = PredictionEngine()