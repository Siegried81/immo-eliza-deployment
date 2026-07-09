import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import xgboost as xgb
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from monitoring.monitor import calculate_psi

# Configure logging for drift warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Define the project root directory.
This allows the script to correctly locate model files
regardless of the execution location.
"""
BASE_DIR = Path(__file__).resolve().parents[1]


"""
Define the source directory containing feature engineering functions.
"""
SRC_DIR = BASE_DIR / "src"


"""
Add the source directory to the Python path.
This allows importing internal project modules.
"""
sys.path.insert(0, str(SRC_DIR))


"""
Import the feature engineering function used to create
the same features as during model training.
"""
from features import add_features

"""
Import monitoring logic for drift detection.
"""
from monitoring.monitor import calculate_psi


"""
Define paths for the trained model and preprocessing pipeline.
"""
MODEL_PATH = BASE_DIR / "models" / "best_XGBoost.json"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"


"""
Custom exception class used for handling prediction failures.
"""
class PredictionError(Exception):
    """Raised when prediction fails"""
    pass


"""
Prediction engine class.

This class:
- Loads the trained XGBoost model and preprocessor
- Transforms new property data
- Checks for data drift
- Generates price predictions
"""
class PredictionEngine:

    """
    Initialize the prediction engine.

    The model and preprocessor are loaded only once
    when the application starts.
    """
    def __init__(self):
        try:
            # Check that the model and preprocessor files exist
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            if not PREPROCESSOR_PATH.exists():
                raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

            # Initialize and load the XGBoost regressor
            self.model = xgb.XGBRegressor()
            self.model.load_model(MODEL_PATH)

            # Load the preprocessing pipeline
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            
        except Exception as e:
            raise PredictionError(f"Failed to initialize prediction engine: {str(e)}")

    """
    Internal helper to check for data drift.
    """
    def _check_drift(self, df: pd.DataFrame):
        psi_values = calculate_psi(df)
        for feature, psi in psi_values.items():
            if psi > 0.25:
                logger.warning(f"🚨 CRITICAL DRIFT detected in {feature}: PSI={psi:.4f}. Retraining recommended.")
        return True

    """
    Generate a property price prediction.

    Input:
        data: dictionary containing property information

    Returns:
        float: Predicted price in EUR
    """
    def predict(self, data: dict) -> float:
        try:
            # Convert the input dictionary into a DataFrame
            df = pd.DataFrame([data])

            # Create additional features required by the model
            df = add_features(df)

            # Check for data drift before proceeding
            self._check_drift(df)

            # Apply the preprocessing pipeline
            X = self.preprocessor.transform(df)

            # Generate the prediction using the trained model
            prediction = self.model.predict(X)

            # Convert the logarithmic prediction back to the original price scale
            return float(np.expm1(prediction)[0])

        except ValueError as e:
            raise PredictionError(f"Data processing error: {str(e)}")
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")

engine = PredictionEngine()