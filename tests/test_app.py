import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

class TestHealthCheck:
    """
    Test suite for the API health check endpoint.
    """
    def test_health_check_success(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "API running"

class TestPredict:
    """
    Test suite for the prediction endpoint.
    """

    def test_predict_valid_input(self):
        """
        Test prediction with valid input data.
        Ensures the API returns a 200 status code and a valid prediction.
        """
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
        
        # Print the actual error message returned by the server if validation fails
        if response.status_code != 200:
            print("\nAPI Error Response:", response.json())
            
        assert response.status_code == 200

    def test_predict_invalid_postcode(self):
        """
        Test prediction with an invalid postal code to trigger validation error.
        Ensures the API returns a 422 status code for out-of-range input.
        """
        payload = {
            "postcode": 999,
            "province": "Brussels",
            "city": "Brussels",
            "property_type": "APARTMENT",
            "livable_surface": 85,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422