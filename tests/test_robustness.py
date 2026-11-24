"""
Tests for failure paths of the service: missing configuration, unexpected
errors, the passing of the calendar year, and the shared quantile-model setup.

Each one pins a behaviour that used to be wrong without any test noticing:
a silent fallback threshold, exception text sent to the client, a build-year
bound frozen at 2026, and two hand-copied model configurations.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.app
from api.app import app
from src.features import CATEGORICAL_FEATURES, DERIVED_FEATURES, NUMERIC_FEATURES, add_features
from src.train import (
    DEFAULT_ROUTING_THRESHOLD,
    LUXURY_THRESHOLD_PATH,
    QUANTILE_LOWER,
    QUANTILE_UPPER,
    quantile_regressor,
)

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


class TestRoutingWithoutLuxuryModel:
    def test_a_training_run_without_a_luxury_model_routes_nothing_to_luxury(self, tmp_path):
        """train.py records luxury_model_available=false when it had too few luxury
        rows; the stale luxury file left behind must not price anything."""
        path = tmp_path / "luxury_threshold.json"
        path.write_text('{"routing_probability_threshold": 0.9, "luxury_model_available": false}')
        threshold = api.app.load_routing_threshold(path)
        result = {"prediction": 300_000.0, "luxury_prediction": 9e9, "luxury_proba": 1.0,
                  "prediction_interval": {"lower": 1.0, "upper": 2.0}}
        assert api.app.route(result, threshold)["segment"] == "standard"


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


class TestOmittedBedroomCount:
    """A request without bedroom_count is imputed like a training row without one."""

    def test_omitted_bedroom_count_is_missing_not_one(self):
        assert api.app.PropertyInput(**APARTMENT).bedroom_count is None

    def test_omitted_bedroom_count_is_priced_at_the_training_median(self):
        import joblib

        from api.predict import MODEL_DIR
        from src.features import NUMERIC_FEATURES

        imputer = (joblib.load(MODEL_DIR / "pipeline.joblib").named_steps["preprocessor"]
                   .named_transformers_["num"].named_steps["imputer"])
        median = int(imputer.statistics_[NUMERIC_FEATURES.index("bedroom_count")])
        omitted = client.post("/predict", json=APARTMENT).json()["prediction"]
        at_median = client.post("/predict", json={**APARTMENT, "bedroom_count": median}).json()["prediction"]
        assert omitted == at_median


class TestUnexpectedValueError:
    def test_a_library_value_error_is_a_500_not_a_422(self, monkeypatch):
        """Only an unknown category is the caller's fault. Any other ValueError
        raised while predicting is a server fault and must not leak its text."""
        def broken(data):
            raise ValueError("Input X contains NaN (sklearn internals)")
        monkeypatch.setattr(api.app.engine, "predict", broken)
        response = client.post("/predict", json=APARTMENT)
        assert response.status_code == 500
        assert "sklearn" not in response.text


class TestInputBounds:
    @pytest.mark.parametrize("field, value", [
        ("livable_surface", 10**15), ("bedroom_count", 10**6), ("total_surface", 10**9),
        ("train_station_distance_m", 10**7), ("terrace", 10**9), ("city", "x" * 100_000),
        ("province", "x" * 1000),
    ])
    def test_absurd_values_are_rejected(self, field, value):
        """An absurd value used to get a price (and a 100 kB line in the log)."""
        assert client.post("/predict", json={**APARTMENT, field: value}).status_code == 422

    def test_the_largest_training_values_are_still_accepted(self):
        biggest = {**APARTMENT, "livable_surface": 5000, "total_surface": 178_367, "bedroom_count": 100,
                   "train_station_distance_m": 13_500, "city": "x" * 30}
        assert client.post("/predict", json=biggest).status_code == 200


class TestRangeContainsTheEstimate:
    def test_the_range_is_widened_to_include_the_point(self, monkeypatch):
        """Three separate models can disagree: a point below the 10th-percentile
        model's value must still sit inside the published range."""
        engine = api.app.engine
        monkeypatch.setattr(engine.pipeline, "predict", lambda df: np.array([np.log1p(100_000.0)]))
        monkeypatch.setattr(engine.pipeline_lower, "predict", lambda df: np.array([np.log1p(150_000.0)]))
        monkeypatch.setattr(engine.pipeline_upper, "predict", lambda df: np.array([np.log1p(200_000.0)]))
        interval = engine.predict(APARTMENT)["prediction_interval"]
        assert interval["lower"] == pytest.approx(100_000.0)
        assert interval["upper"] > 200_000.0


class TestSchemaMatchesModels:
    def test_every_schema_field_is_read_by_a_model(self):
        """A required field no model reads (postcode once) misleads the caller;
        a model input the schema lacks is always imputed."""
        schema = set(api.app.PropertyInput.model_fields) - {"garden_m2"}  # deprecated, ignored
        model_inputs = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(DERIVED_FEATURES)
        assert model_inputs - schema == set()
        assert schema - model_inputs == set()


class TestFlagsAreFoldedToTraining:
    def test_garage_and_terrace_counts_become_yes(self):
        import pandas as pd
        df = add_features(pd.DataFrame([{"livable_surface": 100, "garage": 3, "terrace": 300},
                                        {"livable_surface": 100, "garage": 0, "terrace": None}]))
        assert df["garage"].tolist() == [1, 0]
        assert df["terrace"].iloc[0] == 1.0
        assert pd.isna(df["terrace"].iloc[1])  # unknown stays unknown for the imputer


class TestLuxuryCap:
    """The luxury model only learned prices above ~€1.95M; a €1M house routed to
    it used to get about twice its price."""

    def result(self, standard, luxury, proba):
        return {"prediction": standard, "luxury_prediction": luxury, "luxury_proba": proba,
                "prediction_interval": {"lower": 1.0, "upper": 2.0}}

    def test_a_routed_price_is_capped_by_the_standard_estimate(self):
        from src.train import LUXURY_MAX_RATIO
        chosen = api.app.route(self.result(1_000_000.0, 2_300_000.0, 0.95), 0.9)
        assert chosen["segment"] == "luxury"
        assert chosen["prediction"] == pytest.approx(LUXURY_MAX_RATIO * 1_000_000.0)

    def test_a_luxury_price_under_the_cap_is_kept(self):
        chosen = api.app.route(self.result(2_200_000.0, 2_700_000.0, 0.95), 0.9)
        assert chosen["prediction"] == 2_700_000.0

    def test_below_the_threshold_the_standard_price_is_served(self):
        chosen = api.app.route(self.result(1_000_000.0, 2_300_000.0, 0.5), 0.9)
        assert chosen["prediction"] == 1_000_000.0 and chosen["segment"] == "standard"

    def test_the_rule_works_on_arrays_for_the_evaluation_scripts(self):
        from src.train import routed_price
        out = routed_price(np.array([1e6, 1e6]), np.array([3e6, 3e6]), np.array([0.95, 0.1]), 0.9)
        assert out.tolist() == [1.5e6, 1e6]
