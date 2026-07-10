import os
import sys
import uvicorn

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from monitoring.monitor import log_prediction
from api.predict import engine

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Immo Eliza Price Predictor API",
    version="1.0",
    description="Belgian property price prediction API with XGBoost"
)


class PropertyInput(BaseModel):
    # Location
    postcode: int = Field(..., ge=1000, le=9999, description="Must be between 1000 and 9999")
    province: str
    city: str

    # Property information
    property_type: Literal["HOUSE", "APARTMENT"]
    property_state: Literal[
        "NEW", "EXCELLENT", "FULLY_RENOVATED", "UNDER_CONSTRUCTION",
        "NORMAL", "TO_RENOVATE", "TO_RESTORE", "TO_DEMOLISH"
    ] = "NORMAL"

    livable_surface: int = Field(..., gt=0, description="Livable surface must be greater than 0")
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

    # Cross-field validation to ensure logical data
    @field_validator('total_surface')
    @classmethod
    def total_surface_must_be_ge_livable(cls, v, info):
        if 'livable_surface' in info.data and v > 0 and v < info.data['livable_surface']:
            raise ValueError('Total surface cannot be smaller than livable surface')
        return v


@app.get("/")
def health_check():
    return "alive"


@app.get("/ping")
def ping():
    # Used by UptimeRobot to keep the Render free-tier service awake.
    return {"status": "ok", "message": "pong"}


@app.post("/predict")
def predict(property_input: PropertyInput):
    try:
        data = property_input.model_dump()

        # Generate prediction. engine.predict() returns a dict:
        result = engine.predict(data)

        # Log for monitoring
        log_prediction(data, result["prediction"])

        response = {
            "prediction": round(float(result["prediction"]), 2),
            "currency": "EUR",
            "status": "success",
            "message": "Prediction generated successfully",
            "segment": result.get("segment", "standard"),
        }
        # Only included when the interval models are available.
        if result.get("lower") is not None and result.get("upper") is not None:
            response["lower"] = round(float(result["lower"]), 2)
            response["upper"] = round(float(result["upper"]), 2)

        return response

    except ValueError as ve:
        # Specific validation errors caught
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        import traceback
        print("ERROR DETECTED")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port)
