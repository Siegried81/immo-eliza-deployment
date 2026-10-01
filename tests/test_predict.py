"""
Tests for input normalization and for what the real models return.

The model tests load the committed artifacts and pin relationships rather than
exact prices, so they survive a retraining but catch the encoder silently
ignoring a feature again (a house and an apartment once got the same price).
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.app
from api.app import app
from api.predict import MODEL_DIR, engine
from src.features import load_vocabulary, to_training_vocabulary
from src.train import conformal_margin

client = TestClient(app)
VOCABULARY = load_vocabulary()

HOUSE_IN_LIEGE = {
    "postcode": 4000, "province": "Liège", "city": "Liège",
    "property_type": "HOUSE", "property_state": "NORMAL",
    "livable_surface": 150, "total_surface": 300, "bedroom_count": 3,
    "build_year": 1970, "garage": 1, "garden_m2": 0, "terrace": 10,
    "swimming_pool": False, "energy_consumption_kWh_m2_year": 0,
    "preschool_distance_m": 500, "train_station_distance_m": 800,
    "supermarket_distance_m": 400,
}


class TestToTrainingVocabulary:
    def test_ui_spellings_map_to_training_values(self):
        data, notes = to_training_vocabulary(
            {"property_type": "HOUSE", "property_state": "TO_RENOVATE",
             "province": "Walloon Brabant", "city": "nivelles"}, VOCABULARY)
        assert data["property_type"] == "house"
        assert data["property_state"] == "TO RENOVATE"
        assert data["province"] == "brabant-wallon"
        assert data["city"] == "Nivelles"
        assert notes == []

    @pytest.mark.parametrize("name", ["Liège", "liege", "Luik", "LIEGE"])
    def test_province_aliases(self, name):
        data, _ = to_training_vocabulary(
            {"property_type": "house", "province": name, "city": "Liege"}, VOCABULARY)
        assert data["province"] == "liege"

    def test_is_idempotent(self):
        once, _ = to_training_vocabulary(HOUSE_IN_LIEGE, VOCABULARY)
        twice, _ = to_training_vocabulary(once, VOCABULARY)
        assert once == twice

    def test_unknown_city_is_kept_with_a_note(self):
        data, notes = to_training_vocabulary(
            {"property_type": "house", "province": "Namur", "city": "Atlantis"}, VOCABULARY)
        assert data["city"] == "Atlantis"
        assert "does not appear in the training data" in notes[0]

    def test_rare_city_gets_a_note(self):
        cities = VOCABULARY["provinces"]["namur"]["cities"]
        rare = next(city for city, n in cities.items() if n < 5)
        _, notes = to_training_vocabulary(
            {"property_type": "house", "province": "Namur", "city": rare}, VOCABULARY)
        assert "carries little weight" in notes[0]

    @pytest.mark.parametrize("field, value", [
        ("province", "Utrecht"), ("property_type", "castle"), ("property_state", "HAUNTED"),
    ])
    def test_unknown_category_raises(self, field, value):
        with pytest.raises(ValueError):
            to_training_vocabulary({**HOUSE_IN_LIEGE, field: value}, VOCABULARY)


class TestIntervalCalibration:
    def test_conformal_margin_reaches_target_coverage(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=2000)
        lower, upper = np.full(2000, -0.5), np.full(2000, 0.5)  # too narrow: ~38% coverage
        margin = conformal_margin(lower, upper, y, 0.8)
        assert margin > 0
        assert np.mean((y >= lower - margin) & (y <= upper + margin)) >= 0.8

    def test_engine_applies_the_trained_margin(self):
        calibration = json.loads((MODEL_DIR / "interval_calibration.json").read_text())
        assert engine.interval_margin == calibration["log_margin"]
        assert calibration["check_coverage_calibrated"] > calibration["check_coverage_raw"]


class TestModelUsesCategories:
    def test_house_and_apartment_differ(self):
        house = engine.predict(HOUSE_IN_LIEGE)["prediction"]
        apartment = engine.predict({**HOUSE_IN_LIEGE, "property_type": "APARTMENT"})["prediction"]
        assert house != apartment

    def test_province_changes_the_price(self):
        liege = engine.predict(HOUSE_IN_LIEGE)["prediction"]
        brabant = engine.predict({**HOUSE_IN_LIEGE, "province": "Walloon Brabant",
                                  "city": "Wavre"})["prediction"]
        assert brabant > liege

    def test_luxury_model_is_not_constant(self):
        """It was once trained on prices clipped at the luxury threshold and
        returned 1.9M for every property."""
        villa = {**HOUSE_IN_LIEGE, "province": "Brussels", "city": "Ukkel", "livable_surface": 600,
                 "bedroom_count": 6, "swimming_pool": True}
        small = engine.predict(villa)["luxury_prediction"]
        large = engine.predict({**villa, "livable_surface": 1200, "bedroom_count": 9})["luxury_prediction"]
        assert large != small

    def test_to_renovate_is_cheaper_than_normal(self):
        normal = engine.predict(HOUSE_IN_LIEGE)["prediction"]
        to_renovate = engine.predict({**HOUSE_IN_LIEGE, "property_state": "TO_RENOVATE"})["prediction"]
        assert to_renovate < normal


class TestPredictEndpoint:
    def test_standard_response_has_interval_and_notes(self):
        body = client.post("/predict", json=HOUSE_IN_LIEGE).json()
        assert body["segment"] == "standard"
        assert body["prediction_interval"]["lower"] < body["prediction_interval"]["upper"]
        assert 0 <= body["luxury_probability"] <= 1
        assert body["notes"] == []

    def test_luxury_response_has_no_interval(self, monkeypatch):
        fake = {"prediction": 900_000.0, "luxury_prediction": 4_000_000.0, "luxury_proba": 0.95,
                "prediction_interval": {"lower": 700_000.0, "upper": 1_100_000.0}, "notes": []}
        monkeypatch.setattr(api.app.engine, "predict", lambda data: fake)
        body = client.post("/predict", json=HOUSE_IN_LIEGE).json()
        assert body["segment"] == "luxury"
        assert body["prediction"] == 4_000_000.0
        assert body["prediction_interval"] is None

    def test_unknown_province_returns_422(self):
        response = client.post("/predict", json={**HOUSE_IN_LIEGE, "province": "Utrecht"})
        assert response.status_code == 422
        assert "Unknown province" in response.json()["detail"]

    def test_ping(self):
        assert client.get("/ping").json() == {"status": "alive"}
