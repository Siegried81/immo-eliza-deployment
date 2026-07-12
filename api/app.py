import os
import sys
import uvicorn
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Literal

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
    postcode: int = Field(..., ge=1000, le=9999)
    province: str
    city: str
    property_type: Literal["HOUSE", "APARTMENT"]
    property_state: str = "NORMAL"
    livable_surface: int = Field(..., gt=0)
    total_surface: int = Field(0, ge=0)
    bedroom_count: int = Field(1, ge=0)
    build_year: int = Field(2000, ge=1800, le=2026)
    garage: int = Field(0, ge=0)
    garden_m2: int = Field(0, ge=0)
    terrace: int = Field(0, ge=0)
    swimming_pool: bool = False
    energy_consumption_kWh_m2_year: int = Field(0, ge=0)
    preschool_distance_m: int = Field(0, ge=0)
    train_station_distance_m: int = Field(0, ge=0)
    supermarket_distance_m: int = Field(0, ge=0)

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.get("/ping")
def ping():
    """Lightweight keep-alive endpoint polled by UptimeRobot to prevent cold starts."""
    return {"status": "alive"}

@app.post("/predict")
def predict(property_input: PropertyInput):
    try:
        data = property_input.model_dump()
        result = engine.predict(data)

        # Routing logic based on probability threshold
        luxury_proba = result.get("luxury_proba", 0.0)
        is_luxury = luxury_proba >= LUXURY_THRESHOLD
        final_prediction = result["luxury_prediction"] if is_luxury else result["prediction"]
        segment = "luxury" if is_luxury else "standard"

        log_prediction(data, final_prediction)

        interval = result["prediction_interval"]

        return {
            "prediction": round(float(final_prediction), 2),
            "prediction_interval": {
                "lower": round(interval["lower"], 2),
                "upper": round(interval["upper"], 2)
            },
            "currency": "EUR",
            "status": "success",
            "segment": segment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)