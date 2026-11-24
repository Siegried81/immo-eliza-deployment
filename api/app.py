"""
FastAPI service for Immo Eliza price predictions.

Validates the request, runs the standard, luxury, routing and quantile models
through api.predict.engine, picks the segment with the routing threshold from
models/luxury_threshold.json and logs every prediction for drift monitoring.
"""
import logging
import os
from datetime import date
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from api.predict import THRESHOLD_PATH, engine, load_routing_threshold, route
from monitoring.monitor import log_prediction
from src.features import UnknownCategory

logger = logging.getLogger(__name__)

def current_year() -> int:
    """Upper bound for build_year; a function so tests can move the calendar."""
    return date.today().year


LUXURY_THRESHOLD = load_routing_threshold(THRESHOLD_PATH)

app = FastAPI(title="Immo Eliza API", version="1.0")

class PropertyInput(BaseModel):
    """
    One property, in the units of the training data.

    garage and terrace were 0/1 flags in training, so any positive count or
    area acts as "has one". Optional fields left out are imputed exactly as in
    training (total surface from livable surface, distances by the pipeline's
    median imputer); defaulting them to 0 sent values the model never saw.

    Upper bounds sit well above the largest training value (livable surface
    5,000 m², plot 178,367 m², 100 bedrooms, distances 13.5 km, city names 30
    characters): they turn an absurd request into a 422 instead of a price, and
    cap the size of the line each request adds to the prediction log.
    """
    postcode: int = Field(..., ge=1000, le=9999)
    province: str = Field(..., max_length=64)
    city: str = Field(..., max_length=64)
    # Case is normalized before prediction (see src.features.to_training_vocabulary).
    property_type: Literal["HOUSE", "APARTMENT", "house", "apartment"]
    property_state: str = Field("NORMAL", max_length=64)
    livable_surface: int = Field(..., gt=0, le=20_000)
    total_surface: Optional[int] = Field(None, ge=0, le=1_000_000, description="Plot surface in m²; 0 or missing means unknown.")
    # Missing, not 1: a left-out count goes to the pipeline's median imputer
    # (3 bedrooms), as a training row without one does.
    bedroom_count: Optional[int] = Field(None, ge=0, le=200)
    # Missing, not 2000: a left-out year goes to the pipeline's median imputer,
    # like a training row without one (src/features.py::add_features).
    build_year: Optional[int] = Field(None, ge=1800)
    garage: int = Field(0, ge=0, le=100_000, description="Treated as yes/no: any value above 0 means a garage.")
    garden_m2: int = Field(0, ge=0, le=10_000_000, deprecated=True,
                         description="Ignored: no model reads it. Kept so existing clients that send it still validate.")
    # Missing, not 0: 5.7% of the training listings don't say, and the median
    # imputer filled them with 1 (a terrace). Send 0 for "no terrace".
    terrace: Optional[int] = Field(None, ge=0, le=100_000, description="Treated as yes/no: any value above 0 means a "
                                                           "terrace, 0 means none. Left out, it is unknown.")
    swimming_pool: bool = False
    energy_consumption_kWh_m2_year: int = Field(
        0, ge=0, le=100_000_000, description="kWh/m2/year. 0, or a value above 1000, is read as unknown: "
                             "the typical figure for the property state is used.")
    preschool_distance_m: Optional[int] = Field(None, ge=0, le=200_000)
    train_station_distance_m: Optional[int] = Field(None, ge=0, le=200_000)
    supermarket_distance_m: Optional[int] = Field(None, ge=0, le=200_000)

    @field_validator("total_surface")
    @classmethod
    def zero_surface_is_unknown(cls, value):
        """No training row has a total surface of 0: it can only mean "not filled in"."""
        return None if value == 0 else value

    @field_validator("build_year")
    @classmethod
    def not_in_the_future(cls, value):
        """Checked against today's date, not a year frozen into the schema."""
        if value is not None and value > current_year():
            raise ValueError(f"build_year cannot be after {current_year()}")
        return value

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
    except UnknownCategory as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal error while computing the estimate.") from None

    try:
        chosen = route(result, LUXURY_THRESHOLD)
        log_prediction(data, chosen["prediction"], features=result.get("features"))
        interval = chosen["prediction_interval"]
        likely = chosen["likely_range"]

        return {
            "prediction": round(float(chosen["prediction"]), 2),
            "prediction_interval": None if interval is None else {
                "lower": round(interval["lower"], 2),
                "upper": round(interval["upper"], 2)
            },
            "likely_range": None if likely is None else {
                "lower": round(likely["lower"], 2),
                "upper": round(likely["upper"], 2)
            },
            "currency": "EUR",
            "status": "success",
            "segment": chosen["segment"],
            "luxury_probability": round(result["luxury_proba"], 3),
            "notes": result["notes"],
        }
    except Exception:
        # The traceback goes to the server log; the client gets no internals.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal error while computing the estimate.") from None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
