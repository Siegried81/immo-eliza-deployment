"""
Runs the real src.train.train() on a small synthetic frame, written to a temp
folder, and checks what each model was fitted on.

The other tests read the committed models/, so they cannot see a change to
train() itself. Three past bugs live in that function: the luxury regressor fed
the clipped target (every luxury home priced at the same €1.9M), the clip
bounds taken on all rows instead of the training split, and the
luxury_model_available flag the server relies on.
"""
import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

import src.train as train_module
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features


def synthetic_listings(n, seed=0):
    """Engineered rows shaped like load_dataset()'s output, priced from the surface."""
    rng = np.random.default_rng(seed)
    surface = rng.uniform(40, 400, n).round()
    raw = pd.DataFrame({
        "property_type": rng.choice(["house", "apartment"], n),
        "province": rng.choice(["liege", "brussels"], n),
        "city": rng.choice(["Liege", "Brussels", "Ukkel"], n),
        "property_state": "NORMAL",
        "postcode": rng.choice([1000, 1180, 4000], n),
        "latitude": rng.uniform(50.5, 50.9, n), "longitude": rng.uniform(4.3, 5.6, n),
        "build_year": rng.integers(1900, 2024, n).astype(float),
        "bedroom_count": rng.integers(1, 6, n).astype(float),
        "livable_surface": surface, "total_surface": surface * 2,
        "garage": rng.integers(0, 2, n), "terrace": rng.integers(0, 2, n).astype(float),
        "swimming_pool": rng.integers(0, 2, n),
        "energy_consumption_kWh_m2_year": rng.uniform(80, 400, n),
        "preschool_distance_m": 500.0, "train_station_distance_m": 2000.0,
        "supermarket_distance_m": 700.0,
        "price": np.exp(np.log(surface * 2500) + rng.normal(0, 0.3, n)),
    })
    df = add_features(raw)
    df["area_latitude"], df["area_longitude"] = df["latitude"], df["longitude"]
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["latitude", "longitude", "price"]]


class FitSpy(XGBRegressor):
    """A small, fast XGBRegressor that records every target it is fitted on."""
    targets = []

    def __init__(self, **kwargs):
        kwargs["n_estimators"] = 20
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        FitSpy.targets.append(np.asarray(y, dtype=float))
        return super().fit(X, y, **kwargs)


@pytest.fixture
def run_training(tmp_path, monkeypatch):
    """train() on `n` synthetic rows, every artefact written under tmp_path."""
    def run(n):
        FitSpy.targets = []
        frame = synthetic_listings(n)
        monkeypatch.setattr(train_module, "load_dataset", lambda: frame)
        monkeypatch.setattr(train_module, "XGBRegressor", FitSpy)
        monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)
        monkeypatch.setattr(train_module, "LOCATIONS_PATH", tmp_path / "locations.json")
        for name in ["PIPELINE_PATH", "PIPELINE_LUXURY_PATH", "LUXURY_CLASSIFIER_PATH", "LUXURY_THRESHOLD_PATH",
                     "PIPELINE_LOWER_PATH", "PIPELINE_UPPER_PATH", "INTERVAL_CALIBRATION_PATH"]:
            monkeypatch.setattr(train_module, name, tmp_path / getattr(train_module, name).name)
        train_module.train()
        config = json.loads((tmp_path / "luxury_threshold.json").read_text())
        return frame, config
    return run


def test_luxury_model_learns_unclipped_prices_and_bounds_come_from_training_rows(run_training):
    frame, config = run_training(4000)
    train_rows = train_module.split_all(frame)[0]

    # Bounds from the training split only.
    assert config["luxury_price_threshold"] == pytest.approx(train_module.price_bounds(train_rows["price"])[1])
    assert config["luxury_model_available"] is True

    # The standard model's target is clipped at the luxury price, so the last
    # regressor fitted (the luxury one) must see prices above it, and not one value.
    luxury_target = np.expm1(FitSpy.targets[-1])
    assert luxury_target.min() > config["luxury_price_threshold"]
    assert len(np.unique(luxury_target.round())) > 1
    standard_target = np.expm1(FitSpy.targets[0])
    assert standard_target.max() <= config["luxury_price_threshold"] * (1 + 1e-9)


def test_too_few_luxury_rows_records_that_no_luxury_model_exists(run_training):
    _, config = run_training(600)  # about 4 training rows above the 99th percentile
    assert config["luxury_model_available"] is False
