import sys
import uvicorn
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from monitoring.monitor import log_prediction
from api.predict import engine

"""
Add the project root directory to the Python path.
This allows importing internal project modules.
"""
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


"""
Initialize the FastAPI application.

This API provides Belgian property price predictions
using a trained XGBoost regression model.
"""
app = FastAPI(
    title="Immo Eliza Price Predictor API",
    version="1.0",
    description="Belgian property price prediction API using XGBoost"
)


"""
List of accepted property states.
These values must match the categories used during model training.
"""
PROPERTY_STATES = [
    "NEW",
    "EXCELLENT",
    "FULLY_RENOVATED",
    "UNDER_CONSTRUCTION",
    "NORMAL",
    "TO_RENOVATE",
    "TO_RESTORE",
    "TO_DEMOLISH"
]


"""
List of supported property types.
"""
PROPERTY_TYPES = [
    "HOUSE",
    "APARTMENT"
]


"""
Define the expected input structure for property predictions.

Pydantic validates:
- data types
- allowed values
- minimum and maximum constraints
"""
class PropertyInput(BaseModel):

    # Location
    postcode: int = Field(..., ge=1000, le=9999)
    province: str
    city: str

    # Property information
    property_type: Literal["HOUSE", "APARTMENT"]
    property_state: Literal[
        "NEW", "EXCELLENT", "FULLY_RENOVATED", "UNDER_CONSTRUCTION",
        "NORMAL", "TO_RENOVATE", "TO_RESTORE", "TO_DEMOLISH"
    ] = "NORMAL"

    livable_surface: int = Field(..., gt=0)
    total_surface: int = Field(0, ge=0)
    bedroom_count: int = Field(1, ge=0)
    build_year: int = Field(2000, ge=1800, le=2026)
    garage: int = Field(0, ge=0)
    garden_m2: int = Field(0, ge=0)
    terrace: int = Field(0, ge=0)
    swimming_pool: bool = False

    # Energy and distances
    energy_consumption_kWh_m2_year: int = Field(0, ge=0)
    preschool_distance_m: int = Field(0, ge=0)
    train_station_distance_m: int = Field(0, ge=0)
    supermarket_distance_m: int = Field(0, ge=0)

    # Additional features
    nearest_city: str = ""
    nearest_city_distance_km: int = Field(0, ge=0)


"""
Health check endpoint.
"""
@app.get("/")
def health_check():
    return {"status": "API running"}


"""
Keep-Alive endpoint.
Used by external services (like UptimeRobot) to prevent the API from sleeping.
"""
@app.get("/ping")
def ping():
    return {"status": "alive"}


"""
Prediction endpoint.
"""
@app.post("/predict")
def predict(property_input: PropertyInput):
    try:
        # Convert the validated input object into a dictionary
        data = property_input.model_dump()

        # Generate the property price prediction
        prediction = engine.predict(data)

        # Store input data and prediction result for monitoring
        log_prediction(data, prediction)

        return {
            "prediction": round(float(prediction), 2),
            "currency": "EUR",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)