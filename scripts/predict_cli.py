"""
Price one property from the terminal, with the same engine and routing as the API.

Answers are validated by the API's own schema (api.app.PropertyInput), then
go through api.predict (vocabulary mapping, all five models, the trained
routing threshold, the calibrated range), so the CLI cannot answer
differently from the service. A blank optional field takes the API's default,
which for plot surface and distances is "missing", imputed as in training: a
0 there is a value the models never saw.

Run:
    python scripts/predict_cli.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from api.app import PropertyInput  # noqa: E402
from api.predict import THRESHOLD_PATH, engine, load_routing_threshold, route  # noqa: E402

# (key, prompt, required). Order is the order of the questions.
FIELDS = [
    ("postcode", "Postcode", True),
    ("province", "Province", True),
    ("city", "City", True),
    ("property_type", "Type (house/apartment)", True),
    ("property_state", "State (e.g. NORMAL, TO RENOVATE)", False),
    ("livable_surface", "Livable surface (m²)", True),
    ("total_surface", "Plot surface (m²)", False),
    ("bedroom_count", "Bedrooms", False),
    ("build_year", "Build year", False),
    ("garage", "Garage (1/0)", False),
    ("terrace", "Terrace (1/0)", False),
    ("swimming_pool", "Pool (1/0)", False),
    ("energy_consumption_kWh_m2_year", "Energy consumption (kWh/m²/year)", False),
    ("preschool_distance_m", "Distance to preschool (m)", False),
    ("train_station_distance_m", "Distance to train station (m)", False),
    ("supermarket_distance_m", "Distance to supermarket (m)", False),
]
TEXT_FIELDS = {"province", "city", "property_type", "property_state"}


def parse_answers(answers: dict) -> dict:
    """
    Typed answers -> engine input, through the API schema: blank optional
    fields take its defaults and out-of-range values are rejected the same
    way. Raises ValueError (pydantic's ValidationError is one) on bad input.
    """
    data = {}
    for key, prompt, required in FIELDS:
        raw = str(answers.get(key, "")).strip()
        if not raw:
            if required:
                raise ValueError(f"{prompt} is required.")
            continue
        data[key] = raw if key in TEXT_FIELDS else int(raw)
    return PropertyInput(**data).model_dump()


def run_prediction():
    """Ask every field, then print the price, its segment and its range."""
    print("\n--- IMMO ELIZA: PREDICTION CLI ---  (blank = unknown)")
    answers = {key: input(f"{prompt}: ") for key, prompt, _ in FIELDS}
    try:
        result = engine.predict(parse_answers(answers))
    except ValueError as e:
        print(f"Error: {e}")
        return

    chosen = route(result, load_routing_threshold(THRESHOLD_PATH))
    for note in result["notes"]:
        print(f"Note: {note}")
    print(f"ESTIMATED PRICE: {chosen['prediction']:,.0f} € ({chosen['segment']} model)")
    interval = chosen["prediction_interval"]
    if interval:
        print(f"Likely range: {interval['lower']:,.0f} € - {interval['upper']:,.0f} €")


if __name__ == "__main__":
    run_prediction()
