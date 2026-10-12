"""
Tests for the prediction log, the drift report, the shared training helpers,
the CLI and the hyperparameter search.

They pin what used to be wrong: a log rewritten whole on each request and
read back without the encoded features (so a drift report on real traffic
raised KeyError), a generator that deleted the real log, a CLI that sent 0 for
unknown distances and skipped routing, a missing range calibration read as a
zero margin, and a search tuned on a different setup from the shipped model.
Nothing here needs data/, except the generator test, skipped without it.
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import api.app  # noqa: E402
from api.predict import (CALIBRATION_PATH, engine, load_interval_margin,  # noqa: E402
                         route)
from monitoring import check_drift, monitor  # noqa: E402
from scripts.predict_cli import parse_answers  # noqa: E402
from src.features import (CATEGORICAL_FEATURES, MONITORED_FEATURES,  # noqa: E402
                          NUMERIC_FEATURES)
from src.train import (DATA_PATH, SPLIT_SEED, TEST_SIZE, build_preprocessor,  # noqa: E402
                       clipped_log_target, price_tiers, split)

APARTMENT = {"postcode": 1050, "province": "Brussels", "city": "Elsene",
             "property_type": "apartment", "livable_surface": 85}


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "logs.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(path))
    return path


class TestPredictionLog:
    def test_each_prediction_appends_one_line(self, log_path):
        monitor.log_prediction({"a": 1}, 100.0)
        monitor.log_prediction({"a": 2}, 200.0)
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["prediction"] for line in lines] == [100.0, 200.0]

    def test_concurrent_requests_lose_no_entry(self, log_path):
        def write(i):
            for j in range(50):
                monitor.log_prediction({"i": i, "j": j}, float(j))

        threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(monitor.read_logs()) == 400

    def test_a_truncated_last_line_is_skipped(self, log_path):
        monitor.log_prediction({}, 1.0)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write('{"timestamp": "2026-')
        assert len(monitor.read_logs()) == 1

    def test_api_logs_the_features_the_models_read(self, log_path, monkeypatch):
        monkeypatch.setattr(api.app, "log_prediction", monitor.log_prediction)
        response = TestClient(api.app.app).post("/predict", json=APARTMENT)
        assert response.status_code == 200
        (entry,) = monitor.read_logs()
        assert entry["source"] == "api"
        assert set(entry["features"]) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        assert entry["features"]["property_state_encoded"] == 3  # NORMAL
        assert entry["features"]["preschool_distance_m"] is None  # missing, not 0


class TestDriftReport:
    def training_frame(self, n=500):
        rng = np.random.default_rng(0)
        return pd.DataFrame({f: rng.normal(100, 10, n) for f in MONITORED_FEATURES})

    def test_runs_on_logs_written_by_the_api(self, log_path, monkeypatch):
        monkeypatch.setattr(api.app, "log_prediction", monitor.log_prediction)
        client = TestClient(api.app.app)
        for surface in range(60, 120):
            client.post("/predict", json={**APARTMENT, "livable_surface": surface})
        results, sources = check_drift.drift_report(monitor.read_logs(), self.training_frame())
        assert sources == {"api": 60}
        assert set(results) == set(MONITORED_FEATURES)
        # Distances were never sent: reported as no data, not as NaN.
        assert results["preschool_distance_m"]["psi"] is None
        assert check_drift.status(None) == "no live data"
        assert all(r["psi"] is None or np.isfinite(r["psi"]) for r in results.values())

    def test_same_distribution_reads_as_stable(self):
        train = self.training_frame()
        logs = [{"source": "synthetic", "features": row}
                for row in self.training_frame().sample(frac=1, random_state=1).to_dict("records")]
        results, _ = check_drift.drift_report(logs, train)
        assert all(check_drift.status(r["psi"]) == "OK" for r in results.values())

    def test_entries_without_features_are_skipped(self):
        logs = [{"input": {"build_year": 1990}, "prediction": 1.0}]
        results, sources = check_drift.drift_report(logs, self.training_frame())
        assert sources == {}

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="needs data/ (not in git)")
    def test_generator_never_touches_the_real_log(self, log_path, tmp_path):
        from monitoring.generate_logs import populate_logs
        log_path.write_text('{"source": "api"}\n', encoding="utf-8")
        synthetic = tmp_path / "synthetic.jsonl"
        populate_logs(100, synthetic)
        assert log_path.read_text(encoding="utf-8") == '{"source": "api"}\n'
        entries = monitor.read_logs(synthetic)
        assert len(entries) == 100 and {e["source"] for e in entries} == {"synthetic"}


class TestRoutingAndRange:
    RESULT = {"prediction": 300_000.0, "luxury_prediction": 2_500_000.0,
              "prediction_interval": {"lower": 250_000.0, "upper": 360_000.0}}

    def test_standard_keeps_its_range(self):
        chosen = route({**self.RESULT, "luxury_proba": 0.2}, 0.9)
        assert chosen == {"prediction": 300_000.0, "segment": "standard",
                          "prediction_interval": self.RESULT["prediction_interval"]}

    def test_luxury_has_no_range(self):
        chosen = route({**self.RESULT, "luxury_proba": 0.9}, 0.9)
        assert chosen == {"prediction": 2_500_000.0, "segment": "luxury",
                          "prediction_interval": None}

    def test_missing_calibration_fails_instead_of_a_zero_margin(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_interval_margin(tmp_path / "interval_calibration.json")

    def test_engine_uses_the_calibrated_margin(self):
        expected = json.loads(CALIBRATION_PATH.read_text())["log_margin"]
        assert engine.interval_margin == expected > 0


class TestCli:
    def test_blank_optional_fields_take_the_api_defaults(self):
        data = parse_answers({"postcode": "4000", "province": "Liège", "city": "Liege",
                              "property_type": "house", "livable_surface": "150"})
        assert data == api.app.PropertyInput(**{**APARTMENT, "postcode": 4000, "province": "Liège",
                                                 "city": "Liege", "property_type": "house",
                                                 "livable_surface": 150}).model_dump()
        assert data["preschool_distance_m"] is None and data["total_surface"] is None

    def test_a_required_field_left_blank_is_an_error(self):
        with pytest.raises(ValueError):
            parse_answers({"postcode": "4000", "province": "Liège", "city": "Liege",
                           "property_type": "house"})

    def test_cli_and_api_give_the_same_price(self):
        data = parse_answers({k: str(v) for k, v in APARTMENT.items()})
        cli = route(engine.predict(data), api.app.LUXURY_THRESHOLD)
        api_response = TestClient(api.app.app).post("/predict", json=APARTMENT).json()
        assert round(cli["prediction"], 2) == api_response["prediction"]
        assert cli["segment"] == api_response["segment"]


class TestSharedTrainingHelpers:
    def test_preprocessor_reads_the_declared_features(self):
        columns = [cols for _, _, cols in build_preprocessor().transformers]
        assert columns == [NUMERIC_FEATURES, CATEGORICAL_FEATURES]

    def test_only_property_age_is_left_out_of_drift(self):
        assert set(NUMERIC_FEATURES) - set(MONITORED_FEATURES) == {"property_age"}

    def test_split_matches_the_split_training_always_used(self):
        X = pd.DataFrame({"a": range(200)})
        y = pd.Series(range(200)) * 2.0
        old_train, old_val, old_y_train, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
        new_train, new_val = split(X)
        assert new_train.index.equals(old_train.index) and new_val.index.equals(old_val.index)
        assert y.loc[new_train.index].equals(old_y_train)

    def test_clipped_log_target_is_the_training_target(self):
        prices = pd.Series(np.linspace(10_000, 5_000_000, 1000))
        expected = np.log1p(prices.clip(prices.quantile(0.01), prices.quantile(0.99)))
        assert clipped_log_target(prices).equals(expected)

    def test_price_tiers_cover_every_price_once(self):
        tiers = price_tiers(1_900_000)
        assert [(low, high) for _, low, high in tiers] == [
            (0, 1_000_000), (1_000_000, 1_900_000), (1_900_000, float("inf"))]


class TestOptimize:
    class FakeTrial:
        """Answers every suggestion with the low end of its range."""
        def suggest_int(self, name, low, high):
            return low

        def suggest_float(self, name, low, high, log=False):
            return low

    def test_search_scores_the_shipped_setup(self):
        import optimize
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = pd.Series(X[:, 0] + rng.normal(scale=0.1, size=120))
        data = (X[:90], y[:90], X[90:], y[90:])
        score = optimize.objective(self.FakeTrial(), data)
        assert 0 <= score < 1
