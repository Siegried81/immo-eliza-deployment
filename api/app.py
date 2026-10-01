"""
FastAPI service for Immo Eliza price predictions.

Validates the request, runs the standard, luxury, routing and quantile models
through api.predict.engine, picks the segment with the routing threshold from
models/luxury_threshold.json and logs every prediction for drift monitoring.
"""
import os
import sys
import uvicorn
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from monitoring.monitor import log_prediction
from api.predict import engine

# Load luxury routing config
THRESHOLD_PATH = BASE_DIR / "models" / "luxury_threshold.json"
try:
    with open(THRESHOLD_PATH, "r") as f:
        luxury_cfg = json.load(f)
        LUXURY_THRESHOLD = luxury_cfg.get("routing_probability_threshold", 0.6)
except FileNotFoundError:
    LUXURY_THRESHOLD = 0.6

app = FastAPI(title="Immo Eliza API", version="1.0")

class PropertyInput(BaseModel):
    """
    One property, in the units of the training data.

    garage and terrace were 0/1 flags in training, so any positive count or
    area acts as "has one". Optional fields left out are imputed exactly as in
    training (total surface from livable surface, distances by the pipeline's
    median imputer); defaulting them to 0 sent values the model never saw.
    """
    postcode: int = Field(..., ge=1000, le=9999)
    province: str
    city: str
    # Case is normalized before prediction (see src.features.to_training_vocabulary).
    property_type: Literal["HOUSE", "APARTMENT", "house", "apartment"]
    property_state: str = "NORMAL"
    livable_surface: int = Field(..., gt=0)
    total_surface: Optional[int] = Field(None, ge=0, description="Plot surface in m²; 0 or missing means unknown.")
    bedroom_count: int = Field(1, ge=0)
    build_year: int = Field(2000, ge=1800, le=2026)
    garage: int = Field(0, ge=0, description="Treated as yes/no: any value above 0 means a garage.")
    garden_m2: int = Field(0, ge=0, description="Accepted for backward compatibility; the models do not use it.")
    terrace: int = Field(0, ge=0, description="Treated as yes/no: any value above 0 means a terrace.")
    swimming_pool: bool = False
    energy_consumption_kWh_m2_year: int = Field(0, ge=0)
    preschool_distance_m: Optional[int] = Field(None, ge=0)
    train_station_distance_m: Optional[int] = Field(None, ge=0)
    supermarket_distance_m: Optional[int] = Field(None, ge=0)

    @field_validator("total_surface")
    @classmethod
    def zero_surface_is_unknown(cls, value):
        """No training row has a total surface of 0: it can only mean "not filled in"."""
        return None if value == 0 else value

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.get("/ping")
def ping():
    """Lightweight keep-alive endpoint polled by UptimeRobot to prevent cold starts."""
    return {"status": "alive"}

@app.post("/predict")
def predict(property_input: PropertyInput):
    """
    Predict the price of one property.

    The 10th-90th percentile interval comes from the standard-segment quantile models, so it is
    returned only when the standard model is used: next to a luxury-model price
    it would describe a different estimate. Unknown categories return 422.
    """
    data = property_input.model_dump()
    try:
        result = engine.predict(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # Routing logic based on probability threshold
        luxury_proba = result.get("luxury_proba", 0.0)
        is_luxury = luxury_proba >= LUXURY_THRESHOLD
        final_prediction = result["luxury_prediction"] if is_luxury else result["prediction"]
        segment = "luxury" if is_luxury else "standard"

        log_prediction(data, final_prediction)

        interval = result["prediction_interval"]

        return {
            "prediction": round(float(final_prediction), 2),
            "prediction_interval": None if is_luxury else {
                "lower": round(interval["lower"], 2),
                "upper": round(interval["upper"], 2)
            },
            "currency": "EUR",
            "status": "success",
            "segment": segment,
            "luxury_probability": round(luxury_proba, 3),
            "notes": result["notes"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
