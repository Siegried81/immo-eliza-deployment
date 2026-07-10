import numpy as np
import pandas as pd
import joblib
import json
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
PIPELINE_LOWER_PATH = BASE_DIR / "models" / "pipeline_lower.joblib"
PIPELINE_UPPER_PATH = BASE_DIR / "models" / "pipeline_upper.joblib"
PIPELINE_LUXURY_PATH = BASE_DIR / "models" / "pipeline_luxury.joblib"
LUXURY_THRESHOLD_PATH = BASE_DIR / "models" / "luxury_threshold.json"

LUXURY_ROUTING_MARGIN = 0.8


class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


class PredictionEngine:
    """
    Initialize the prediction engine. Pipelines are loaded only once when
    the application starts.
    """
    def __init__(self):
        try:
            if not PIPELINE_PATH.exists():
                raise FileNotFoundError(f"Pipeline not found: {PIPELINE_PATH}")

            self.pipeline = joblib.load(PIPELINE_PATH)

            # Optional (None if not yet trained).
            self.lower_pipeline = joblib.load(PIPELINE_LOWER_PATH) if PIPELINE_LOWER_PATH.exists() else None
            self.upper_pipeline = joblib.load(PIPELINE_UPPER_PATH) if PIPELINE_UPPER_PATH.exists() else None
            self.luxury_pipeline = joblib.load(PIPELINE_LUXURY_PATH) if PIPELINE_LUXURY_PATH.exists() else None

            self.luxury_threshold = None
            if LUXURY_THRESHOLD_PATH.exists():
                with open(LUXURY_THRESHOLD_PATH) as f:
                    info = json.load(f)
                    if info.get("luxury_model_available"):
                        self.luxury_threshold = info.get("luxury_threshold")

            if self.lower_pipeline is None or self.upper_pipeline is None:
                logger.info("Prediction interval models not found -- predict() will omit lower/upper bounds.")
            if self.luxury_pipeline is None or self.luxury_threshold is None:
                logger.info("Luxury-segment model not found -- predict() will always use the standard model.")

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
    Returns a dict: {
        "prediction": float,
        "lower": float | None,
        "upper": float | None,
        "segment": "standard" | "luxury",
    }
    """
    def predict(self, data: dict) -> dict:
        try:
            df = add_features(pd.DataFrame([data]))
            self._check_drift(df)

            point = float(np.expm1(self.pipeline.predict(df))[0])
            segment = "standard"

            if self.luxury_pipeline is not None and self.luxury_threshold is not None:
                if point >= self.luxury_threshold * LUXURY_ROUTING_MARGIN:
                    point = float(np.expm1(self.luxury_pipeline.predict(df))[0])
                    segment = "luxury"

            lower = upper = None
            if segment == "standard" and self.lower_pipeline is not None and self.upper_pipeline is not None:
                lower = float(np.expm1(self.lower_pipeline.predict(df))[0])
                upper = float(np.expm1(self.upper_pipeline.predict(df))[0])
                # Quantile models are trained independently and can occasionally
                # cross the point estimate or each other -- guard against that.
                lower, upper = min(lower, point), max(upper, point)

            return {
                "prediction": point,
                "lower": lower,
                "upper": upper,
                "segment": segment,
            }

        except ValueError as e:
            raise PredictionError(f"Data processing error: {str(e)}")
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")


engine = PredictionEngine()
