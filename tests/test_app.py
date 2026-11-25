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
        assert response.status_code == 200, response.json()
        body = response.json()
        # An 85 m² Brussels apartment: a constant or zero price would pass a
        # status check alone.
        assert 100_000 < body["prediction"] < 1_000_000
        assert body["prediction_interval"]["lower"] <= body["prediction"] <= body["prediction_interval"]["upper"]

    def test_docs_example_is_a_plausible_request(self):
        """
        The example /docs pre-fills is the first request most visitors send.
        It must validate, name a city the models know (no "unknown city"
        note), and come back with an ordinary Brussels apartment price, not a
        value built from the schema's bounds.
        """
        schema = client.get("/openapi.json").json()["components"]["schemas"]["PropertyInput"]
        example = schema["examples"][0]
        response = client.post("/predict", json=example)
        assert response.status_code == 200, response.json()
        body = response.json()
        assert body["notes"] == []
        assert 150_000 < body["prediction"] < 700_000

    def test_a_client_still_sending_garden_m2_gets_its_price(self):
        """
        garden_m2 is out of the schema (no model ever read it), and a client
        written against the old schema still sends it. Pydantic ignores fields
        it does not declare, so that request must keep answering 200 with the
        same price as one without the field.
        """
        schema = client.get("/openapi.json").json()["components"]["schemas"]["PropertyInput"]
        assert "garden_m2" not in schema["properties"]
        example = schema["examples"][0]
        legacy = client.post("/predict", json={**example, "garden_m2": 120})
        assert legacy.status_code == 200, legacy.json()
        assert legacy.json()["prediction"] == client.post("/predict", json=example).json()["prediction"]

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
