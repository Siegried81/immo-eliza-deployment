import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


class TestHealthCheck:
    """
    Test API health endpoint.
    """

    def test_health_check_success(self):
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "API alive"


class TestPredict:
    """
    Test prediction endpoint with valid input data.
    """

    def test_predict_valid_input(self):
        payload = {
            "postcode": 1000,
            "province": "Brussels",
            "city": "Brussels",
            "property_type": "APARTMENT",
            "property_state": "NORMAL",
            "livable_surface": 85,
            "total_surface": 100,
            "bedroom_count": 2,
            "build_year": 2000,
            "garage": 0,
            "garden_m2": 0,
            "terrace": 0,
            "swimming_pool": False,
            "energy_consumption_kWh_m2_year": 0,
            "preschool_distance_m": 500,
            "train_station_distance_m": 800,
            "supermarket_distance_m": 400,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert data["currency"] == "EUR"


    """
    Test prediction endpoint with invalid input data.
    Invalid postcode should trigger validation error.
    """

    def test_predict_invalid_postcode(self):
        payload = {
            "postcode": 999,
            "province": "Brussels",
            "city": "Brussels",
            "property_type": "APARTMENT",
            "livable_surface": 85,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422