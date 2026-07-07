from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from api.predict import engine
from monitoring.monitoring import log_prediction


# =====================================================
# API CONFIGURATION
# =====================================================

app = FastAPI(
    title="Immo Eliza Price Predictor API",
    version="1.0",
    description="Belgian property price prediction API using XGBoost"
)



# =====================================================
# CONSTANTS
# =====================================================

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


PROPERTY_TYPES = [
    "HOUSE",
    "APARTMENT"
]



# =====================================================
# INPUT SCHEMA
# =====================================================

class PropertyInput(BaseModel):

    # -------------------------
    # Location
    # -------------------------

    postcode: int = Field(
        ...,
        ge=1000,
        le=9999
    )

    province: str

    city: str



    # -------------------------
    # Property information
    # -------------------------

    property_type: Literal[
        "HOUSE",
        "APARTMENT"
    ]

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


    livable_surface: float = Field(
        ...,
        gt=0
    )

    total_surface: float = Field(
        0,
        ge=0
    )

    bedroom_count: int = Field(
        1,
        ge=0
    )

    build_year: int = Field(
        2000,
        ge=1800,
        le=2026
    )


    garage: int = Field(
        0,
        ge=0
    )

    garden_m2: float = Field(
        0,
        ge=0
    )

    terrace: float = Field(
        0,
        ge=0
    )

    swimming_pool: bool = False



    # -------------------------
    # Energy & distances
    # -------------------------

    energy_consumption_kWh_m2_year: float = Field(
        0,
        ge=0
    )


    preschool_distance_m: float = Field(
        0,
        ge=0
    )


    train_station_distance_m: float = Field(
        0,
        ge=0
    )


    supermarket_distance_m: float = Field(
        0,
        ge=0
    )



    # -------------------------
    # Additional features
    # -------------------------

    nearest_city: str = ""

    nearest_city_distance_km: float = Field(
        0,
        ge=0
    )



# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def health_check():

    return {
        "status": "API running"
    }



# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
def predict(property_input: PropertyInput):

    try:

        data = property_input.model_dump()


        prediction = engine.predict(
            data
        )


        # Store prediction for monitoring
        log_prediction(
            data,
            prediction
        )


        return {
            "prediction": round(
                prediction,
                2
            ),
            "currency": "EUR",
            "status": "success"
        }



    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )