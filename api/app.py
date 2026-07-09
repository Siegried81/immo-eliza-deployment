import sys
import uvicorn
import os

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Literal
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
    return {"status": "API running"}

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "pong"}

@app.post("/predict")
def predict(property_input: PropertyInput):
    try:
        data = property_input.model_dump()

        # Generate prediction
        prediction = engine.predict(data)

        # Log for monitoring
        log_prediction(data, prediction)

        return {
            "prediction": round(float(prediction), 2),
            "currency": "EUR",
            "status": "success",
            "message": "Prediction generated successfully"
        }

    except ValueError as ve:
        # Specific validation errors caught here
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        import traceback
        print("--- ERROR DETECTED ---")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e))

if __name__ == "__main__":
    # For local development, run this from the root with: python -m api.app
    uvicorn.run("api.app:app", host="0.0.0.0", port=port)
