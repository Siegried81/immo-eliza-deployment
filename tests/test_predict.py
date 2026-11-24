"""
Tests for input normalization and for what the real models return.

The model tests load the committed artifacts and pin relationships rather than
exact prices, so they survive a retraining but catch the encoder silently
ignoring a feature again (a house and an apartment once got the same price).
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.app
from api.app import app
from api.predict import MODEL_DIR, engine
from src.features import CITY_ALIASES, NUMERIC_FEATURES, load_vocabulary, to_training_vocabulary
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
        # "!=" on two floats proves little. The bug it guards against was the
        # encoder mapping the type to the global mean, so check the encoding:
        # the model itself gives the type a small weight once surfaces are set.
        import pandas as pd

        from src.features import add_features, add_location, to_training_vocabulary
        rows = [to_training_vocabulary({**HOUSE_IN_LIEGE, "property_type": t}, engine.vocabulary)[0]
                for t in ("HOUSE", "APARTMENT")]
        features = add_location(add_features(pd.DataFrame(rows)), engine.locations)
        encoder = engine.pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        encoded = np.asarray(encoder.transform(features[["property_type", "province", "city", "property_state"]]))
        assert abs(encoded[0, 0] - encoded[1, 0]) > 0.1

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
        assert large > small

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
        fake = {"prediction": 2_000_000.0, "luxury_prediction": 2_800_000.0, "luxury_proba": 0.95,
                "prediction_interval": {"lower": 700_000.0, "upper": 1_100_000.0}, "notes": []}
        monkeypatch.setattr(api.app.engine, "predict", lambda data: fake)
        body = client.post("/predict", json=HOUSE_IN_LIEGE).json()
        assert body["segment"] == "luxury"
        assert body["prediction"] == 2_800_000.0
        assert body["prediction_interval"] is None

    def test_unknown_province_returns_422(self):
        response = client.post("/predict", json={**HOUSE_IN_LIEGE, "province": "Utrecht"})
        assert response.status_code == 422
        assert "Unknown province" in response.json()["detail"]

    def test_ping(self):
        assert client.get("/ping").json() == {"status": "alive"}


class TestStateEncoding:
    """The two state dictionaries are keyed the way the data spells the states.

    They used to carry underscores, which no row matched, so every multi-word
    state was encoded as NORMAL in training and at inference alike.
    """

    @pytest.mark.parametrize("spelling", ["TO RENOVATE", "to renovate", "to_renovate", " TO  RENOVATE "])
    def test_a_multi_word_state_keeps_its_own_ordinal_and_energy(self, spelling):
        from src.features import add_features

        df = add_features(pd.DataFrame([{**HOUSE_IN_LIEGE, "property_state": spelling}]))
        assert df.loc[0, "property_state"] == "TO RENOVATE"
        assert df.loc[0, "property_state_encoded"] == 2
        assert df.loc[0, "energy_consumption_kWh_m2_year"] == 420

    def test_every_training_state_has_an_ordinal_and_an_energy(self):
        from src.features import ENERGY_MEAN_BY_STATE, STATE_MAPPING

        for state in VOCABULARY["property_states"]:
            assert state in STATE_MAPPING, state
            assert state in ENERGY_MEAN_BY_STATE, state


class TestEnergy:
    """The models read the consumption the listing gives.

    The cleaned data names the column "energy_consumption_kWh/m2/year" and the
    code looked for "energy_consumption_kWh_m2_year", so training never saw a
    real value: every row got the figure for its state.
    """

    def test_the_data_spelling_of_the_energy_column_is_renamed(self):
        from src.features import RAW_COLUMN_NAMES, add_features, to_model_columns

        listings = pd.DataFrame([{**HOUSE_IN_LIEGE, "energy_consumption_kWh_m2_year": 0}])
        raw_name = next(iter(RAW_COLUMN_NAMES))
        listings = listings.drop(columns="energy_consumption_kWh_m2_year").assign(**{raw_name: 264})
        df = add_features(to_model_columns(listings))
        assert df.loc[0, "energy_consumption_kWh_m2_year"] == 264

    def test_listings_without_a_model_input_are_refused(self):
        from src.features import to_model_columns

        listings = pd.DataFrame([HOUSE_IN_LIEGE]).drop(columns="energy_consumption_kWh_m2_year")
        with pytest.raises(KeyError, match="energy_consumption_kWh_m2_year"):
            to_model_columns(listings)

    @pytest.mark.parametrize("given, read", [(264, 264), (1000, 1000), (0, 260), (None, 260),
                                             (1001, 260), (20_260_607, 260)])
    def test_an_implausible_consumption_counts_as_not_given(self, given, read):
        from src.features import add_features

        df = add_features(pd.DataFrame([{**HOUSE_IN_LIEGE, "energy_consumption_kWh_m2_year": given}]))
        assert df.loc[0, "energy_consumption_kWh_m2_year"] == read

    def test_a_frugal_house_is_not_priced_below_an_energy_hungry_one(self):
        frugal = engine.predict({**HOUSE_IN_LIEGE, "energy_consumption_kWh_m2_year": 100})["prediction"]
        hungry = engine.predict({**HOUSE_IN_LIEGE, "energy_consumption_kWh_m2_year": 500})["prediction"]
        assert frugal > hungry

    def test_a_typical_consumption_does_not_read_as_a_ruin(self):
        """250 kWh is the median listing. On the old models it sat between the
        codes for "to restore" (220) and "to demolish" (300) and cost 16%. A
        consumption left out is now filled with the state's median (260 for
        NORMAL), so the two requests are nearly the same house."""
        unknown = engine.predict({**HOUSE_IN_LIEGE, "energy_consumption_kWh_m2_year": 0})["prediction"]
        typical = engine.predict({**HOUSE_IN_LIEGE, "energy_consumption_kWh_m2_year": 250})["prediction"]
        assert abs(typical / unknown - 1) < 0.05


class TestLocation:
    """The models read where the postcode is, from centroids of the training rows."""

    LISTINGS = pd.DataFrame({
        "province": ["liege", "liege", "liege", "east-flanders"],
        "city": ["Liege", "Liege", "Liege", "Gent"],
        "postcode": [4000, 4000, 4020, 9000],
        "latitude": [50.64, 50.66, 50.60, 51.05],
        "longitude": [5.56, 5.58, 5.60, 3.72],
    })

    def test_centroids_are_medians_of_the_listings_given(self):
        from src.features import build_locations

        locations = build_locations(self.LISTINGS)
        assert locations["postcodes"]["4000"] == [50.65, 5.57]
        assert locations["cities"]["liege"]["Liege"] == [50.64, 5.58]
        assert locations["city_postcodes"] == {"liege": {"Liege": 4000}, "east-flanders": {"Gent": 9000}}

    def test_a_city_name_in_two_provinces_is_two_places(self):
        """Halle is a town near Brussels and a village near Antwerp."""
        from src.features import add_location, build_locations

        listings = pd.DataFrame({
            "province": ["vlaams-brabant", "vlaams-brabant", "antwerp"],
            "city": ["Halle", "Halle", "Halle"], "postcode": [1500, 1500, 2980],
            "latitude": [50.73, 50.74, 51.24], "longitude": [4.23, 4.24, 4.65]})
        locations = build_locations(listings)
        assert locations["city_postcodes"]["antwerp"]["Halle"] == 2980
        assert locations["city_postcodes"]["vlaams-brabant"]["Halle"] == 1500
        request = pd.DataFrame([{"postcode": None, "province": "antwerp", "city": "Halle"}])
        assert add_location(request, locations).loc[0, "area_latitude"] == 51.24

    def test_a_city_filed_under_another_province_still_has_a_place(self):
        from src.features import add_location, build_locations

        request = pd.DataFrame([{"postcode": None, "province": "namur", "city": "Gent"}])
        assert add_location(request, build_locations(self.LISTINGS)).loc[0, "area_latitude"] == 51.05

    @pytest.mark.parametrize("postcode, city, expected", [
        (4020, "Liege", (50.60, 5.60)),      # the postcode's own centroid
        (4031, "Liege", (50.64, 5.58)),      # unknown postcode: the city's
        (None, "Gent", (51.05, 3.72)),       # no postcode: the city's
    ])
    def test_postcode_first_then_city(self, postcode, city, expected):
        from src.features import add_location, build_locations

        df = add_location(pd.DataFrame([{"postcode": postcode, "city": city}]), build_locations(self.LISTINGS))
        assert (df.loc[0, "area_latitude"], df.loc[0, "area_longitude"]) == expected

    def test_an_unknown_place_is_missing_not_zero(self):
        from src.features import add_location, build_locations

        df = add_location(pd.DataFrame([{"postcode": 1234, "city": "Nowhere"}]), build_locations(self.LISTINGS))
        assert df[["area_latitude", "area_longitude"]].isna().all(axis=None)

    def test_every_listed_city_has_a_centroid_and_a_postcode(self):
        from src.features import load_locations

        locations = load_locations()
        for slug, province in VOCABULARY["provinces"].items():
            for city in province["cities"]:
                assert city in locations["cities"][slug], city
                assert 1000 <= locations["city_postcodes"][slug][city] <= 9999

    def test_the_postcode_changes_the_price(self):
        """Same city, same house: Sart-Tilman (4031) and the centre (4000) are
        not the same market. The models used to ignore the postcode."""
        centre = engine.predict({**HOUSE_IN_LIEGE, "postcode": 4000})["prediction"]
        other = engine.predict({**HOUSE_IN_LIEGE, "postcode": 4020})["prediction"]
        assert centre != other

    def test_a_postcode_outside_the_training_data_gets_a_note(self):
        result = engine.predict({**HOUSE_IN_LIEGE, "postcode": 9999})
        assert any("postcode 9999" in note for note in result["notes"])
        assert not any("postcode" in note for note in engine.predict(HOUSE_IN_LIEGE)["notes"])


class TestUnknownFields:
    """A field the caller leaves out is priced like a training row without it."""

    def test_a_left_out_terrace_is_unknown_not_absent(self):
        """Training filled an unknown terrace with the median, 1. The API used
        to send 0, so the same listing was priced up to 13% lower by the API
        than by the pipeline that was scored."""
        request = {k: v for k, v in HOUSE_IN_LIEGE.items() if k != "terrace"}
        assert api.app.PropertyInput(**request).terrace is None
        left_out = client.post("/predict", json=request).json()["prediction"]
        with_one = client.post("/predict", json={**request, "terrace": 1}).json()["prediction"]
        assert left_out == with_one

    def test_engine_alone_fills_a_missing_surface_or_terrace_like_training(self):
        from src.features import add_features

        df = add_features(pd.DataFrame([{"property_type": "house", "province": "liege", "city": "Liege"}]))
        assert df[["livable_surface", "terrace", "total_surface"]].isna().all(axis=None)


class TestCityAliases:
    """A city typed in the other national language must reach the model."""

    @pytest.mark.parametrize("typed, province, expected", [
        ("Anvers", "Antwerp", "Antwerp"),
        ("Antwerpen", "Antwerp", "Antwerp"),
        ("Gand", "East Flanders", "Gent"),
        ("Ghent", "East Flanders", "Gent"),
        ("Bruges", "West Flanders", "Brugge"),
        ("Courtrai", "West Flanders", "Kortrijk"),
        ("Luik", "Liège", "Liege"),
        ("Bergen", "Hainaut", "Mons"),
        ("Namen", "Namur", "Namur"),
        ("Doornik", "Hainaut", "Tournai"),
        ("Louvain", "Flemish Brabant", "Leuven"),
        ("Malines", "Antwerp", "Mechelen"),
        ("Waver", "Walloon Brabant", "Wavre"),
        ("Nijvel", "Walloon Brabant", "Nivelles"),
        ("Bruxelles", "Brussels", "Brussels"),
        ("Ixelles", "Brussels", "Elsene"),
        ("Uccle", "Brussels", "Ukkel"),
        ("Woluwe-Saint-Pierre", "Brussels", "Sint-Pieters-Woluwe"),
    ])
    def test_alias_resolves_to_the_training_spelling(self, typed, province, expected):
        data, notes = to_training_vocabulary(
            {"property_type": "house", "province": province, "city": typed}, VOCABULARY)
        assert data["city"] == expected
        assert not any("does not appear in the training data" in n for n in notes)

    def test_every_alias_points_at_a_known_city(self):
        known = {city for info in VOCABULARY["provinces"].values() for city in info["cities"]}
        assert {target for target in CITY_ALIASES.values()} <= known

    def test_ambiguous_alias_prefers_the_users_province(self):
        """"Forest" is Vorst in Brussels and also a Vorst in Antwerp province."""
        data, _ = to_training_vocabulary(
            {"property_type": "house", "province": "Brussels", "city": "Forest"}, VOCABULARY)
        assert data["city"] == "Vorst"

    def test_is_still_idempotent(self):
        once, _ = to_training_vocabulary(
            {"property_type": "house", "province": "Antwerp", "city": "Anvers"}, VOCABULARY)
        twice, _ = to_training_vocabulary(once, VOCABULARY)
        assert once == twice


class TestEngineWithoutOptionalFields:
    """The model card calls engine.predict on a dict with only the required fields."""

    MINIMAL = {"postcode": 1050, "province": "Brussels", "city": "Ixelles",
               "property_type": "APARTMENT", "livable_surface": 90}

    def test_absent_counts_distances_and_year_are_missing_not_zero(self):
        features = engine.predict(self.MINIMAL)["features"]
        for name in ["bedroom_count", "preschool_distance_m", "train_station_distance_m",
                     "supermarket_distance_m", "build_year", "property_age"]:
            assert features[name] is None, name

    def test_engine_and_api_give_the_same_price(self):
        # build_year left out on both sides: one rule, the training median.
        direct = engine.predict(self.MINIMAL)["prediction"]
        served = client.post("/predict", json=self.MINIMAL).json()["prediction"]
        assert round(direct, 2) == served

    def test_a_missing_build_year_is_priced_at_the_training_median(self):
        imputer = engine.pipeline.named_steps["preprocessor"].named_transformers_["num"]
        median_year = imputer.named_steps["imputer"].statistics_[NUMERIC_FEATURES.index("build_year")]
        left_out = engine.predict(self.MINIMAL)["prediction"]
        at_median = engine.predict({**self.MINIMAL, "build_year": int(median_year)})["prediction"]
        assert median_year == int(median_year)
        assert left_out == at_median
