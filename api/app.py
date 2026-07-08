import sys
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

    """
    Belgian postal code.
    Only valid Belgian 4-digit postcodes are accepted.
    """
    postcode: int = Field(
        ...,
        ge=1000,
        le=9999
    )

    """
    Province where the property is located.
    """
    province: str

    """
    City where the property is located.
    """
    city: str


    # Property information

    """
    Type of property.
    Accepted values are HOUSE or APARTMENT.
    """
    property_type: Literal[
        "HOUSE",
        "APARTMENT"
    ]

    """
    Current condition of the property.
    Default value is NORMAL.
    """
    property_state: Literal[
        "NEW",
        "EXCELLENT",
        "FULLY_RENOVATED",
        "UNDER_CONSTRUCTION",
        "NORMAL",
        "TO_RENOVATE",
        "TO_RESTORE",
        "TO_DEMOLISH"
    ] = "NORMAL"

    """
    Living surface area in square meters.
    This value is mandatory and must be greater than zero.
    """
    livable_surface: int = Field(
        ...,
        gt=0
    )

    """
    Total surface area of the property in square meters.
    """
    total_surface: int = Field(
        0,
        ge=0
    )

    """
    Number of bedrooms.
    """
    bedroom_count: int = Field(
        1,
        ge=0
    )

    """
    Year when the property was built.
    """
    build_year: int = Field(
        2000,
        ge=1800,
        le=2026
    )

    """
    Number of garages available.
    """
    garage: int = Field(
        0,
        ge=0
    )

    """
    Garden surface area in square meters.
    """
    garden_m2: int = Field(
        0,
        ge=0
    )

    """
    Indicates whether the property has a terrace.
    """
    terrace: int = Field(
        0,
        ge=0
    )

    """
    Indicates whether the property has a swimming pool.
    """
    swimming_pool: bool = False


    # Energy and distances

    """
    Energy consumption in kWh per square meter per year.
    """
    energy_consumption_kWh_m2_year: int = Field(
        0,
        ge=0
    )

    """
    Distance to the nearest preschool in meters.
    """
    preschool_distance_m: int = Field(
        0,
        ge=0
    )

    """
    Distance to the nearest train station in meters.
    """
    train_station_distance_m: int = Field(
        0,
        ge=0
    )

    """
    Distance to the nearest supermarket in meters.
    """
    supermarket_distance_m: int = Field(
        0,
        ge=0
    )


    # Additional features

    """
    Name of the nearest city.
    """
    nearest_city: str = ""

    """
    Distance to the nearest city in kilometers.
    """
    nearest_city_distance_km: int = Field(
        0,
        ge=0
    )


"""
Health check endpoint.

Used to verify that the API is running correctly.
"""
@app.get("/")
def health_check():

    return {
        "status": "API running"
    }


"""
Prediction endpoint.

Receives property information,
generates a prediction using the trained model,
stores the prediction for monitoring,
and returns the estimated price.
"""


@app.post("/predict")
def predict(property_input: PropertyInput):

    try:

        """
        Convert the validated input object into a dictionary.
        """
        data = property_input.model_dump()

        """
        Generate the property price prediction.
        """
        prediction = engine.predict(
            data
        )

        """
        Store input data and prediction result.
        This allows future monitoring and drift analysis.
        """
        log_prediction(
            data,
            prediction
        )

        """
        Return the prediction result as JSON.
        """
        return {
            "prediction": round(
                float(prediction),
                2
            ),
            "currency": "EUR",
            "status": "success"
        }

    except Exception as e:

        """
        Return an HTTP 400 error if prediction fails.
        """
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )