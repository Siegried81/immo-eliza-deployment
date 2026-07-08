import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from api.app import app

"""
Add the project root directory to the Python path
to allow importing the FastAPI application.
"""
sys.path.insert(0, str(Path(__file__).parent.parent))

client = TestClient(app)

class TestHealthCheck:
    def test_health_check_success(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "API running"

class TestPredict:
    
    """
    Test prediction with valid input data.
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
            "nearest_city": "Brussels",
            "nearest_city_distance_km": 0
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert response.json()["currency"] == "EUR"

    """
    Test prediction with an invalid postal code to trigger validation error.
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