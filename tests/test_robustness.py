"""
Tests for failure paths of the service: missing configuration, unexpected
errors, the passing of the calendar year, and the shared quantile-model setup.

Each one pins a behaviour that used to be wrong without any test noticing:
a silent fallback threshold, exception text sent to the client, a build-year
bound frozen at 2026, and two hand-copied model configurations.
"""
import pytest
from fastapi.testclient import TestClient

import api.app
from api.app import app
from src.train import (DEFAULT_ROUTING_THRESHOLD, LUXURY_THRESHOLD_PATH,
                       QUANTILE_LOWER, QUANTILE_UPPER, quantile_regressor)

client = TestClient(app)

APARTMENT = {
    "postcode": 1050, "province": "Brussels", "city": "Elsene",
    "property_type": "apartment", "livable_surface": 85,
}


class TestRoutingThreshold:
    def test_served_threshold_is_the_trained_one(self):
        assert api.app.LUXURY_THRESHOLD == DEFAULT_ROUTING_THRESHOLD
        assert api.app.load_routing_threshold(LUXURY_THRESHOLD_PATH) == DEFAULT_ROUTING_THRESHOLD

    def test_missing_config_fails_instead_of_guessing(self, tmp_path):
        # A guessed threshold silently changes which model prices a property.
        with pytest.raises(FileNotFoundError):
            api.app.load_routing_threshold(tmp_path / "luxury_threshold.json")


class TestServerErrors:
    def test_500_does_not_send_exception_text(self, monkeypatch):
        # The engine result lacks "prediction_interval": a KeyError inside /predict.
        monkeypatch.setattr(api.app.engine, "predict", lambda data: {
            "prediction": 300_000.0, "luxury_prediction": 2_000_000.0,
            "luxury_proba": 0.1, "notes": [],
        })
        response = client.post("/predict", json=APARTMENT)
        assert response.status_code == 500
        assert "prediction_interval" not in response.text
        assert response.json()["detail"] == "Internal error while computing the estimate."


class TestBuildYearBound:
    def test_a_build_from_the_current_year_is_accepted(self, monkeypatch):
        monkeypatch.setattr(api.app, "current_year", lambda: 2031)
        response = client.post("/predict", json={**APARTMENT, "build_year": 2031})
        assert response.status_code == 200

    def test_a_build_from_next_year_is_rejected(self, monkeypatch):
        monkeypatch.setattr(api.app, "current_year", lambda: 2031)
        response = client.post("/predict", json={**APARTMENT, "build_year": 2032})
        assert response.status_code == 422


class TestQuantileRegressor:
    @pytest.mark.parametrize("alpha", [QUANTILE_LOWER, QUANTILE_UPPER])
    def test_both_bounds_share_one_configuration(self, alpha):
        params = quantile_regressor(alpha).get_params()
        assert params["objective"] == "reg:quantileerror"
        assert params["quantile_alpha"] == alpha
        # The values the committed pipeline_lower/upper.joblib were trained with.
        assert {k: params[k] for k in ("n_estimators", "learning_rate", "max_depth",
                                       "random_state", "early_stopping_rounds")} == {
            "n_estimators": 800, "learning_rate": 0.05, "max_depth": 6,
            "random_state": 42, "early_stopping_rounds": 50,
        }

    def test_committed_models_match_that_configuration(self):
        import joblib
        from api.predict import MODEL_DIR
        for alpha, name in [(QUANTILE_LOWER, "pipeline_lower.joblib"),
                            (QUANTILE_UPPER, "pipeline_upper.joblib")]:
            trained = joblib.load(MODEL_DIR / name).named_steps["model"].get_params()
            expected = quantile_regressor(alpha).get_params()
            # "missing" is NaN on both sides, and NaN never equals itself.
            expected.pop("missing")
            assert {k: trained[k] for k in expected} == expected
