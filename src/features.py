import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Reference year of the training data, not today's date: property_age is
# CURRENT_YEAR - build_year, and the models learned ages counted from 2026.
# Moving it with the calendar would age every property by one year each
# January without a retraining. Update it only together with a retraining.
CURRENT_YEAR = 2026

VOCABULARY_PATH = Path(__file__).resolve().parent.parent / "models" / "categories.json"

# TargetEncoder's default min_samples_leaf (the encoder in src/train.py uses
# the defaults): a city with fewer training rows is pulled mostly towards the
# global mean, so its own price level barely counts.
LOW_SAMPLE_CITY = 20

# Keys are spelled the way the training data and models/categories.json spell
# the states: upper case, words separated by a SPACE. They used to carry an
# underscore ("TO_RENOVATE"), which no row ever matched, so every multi-word
# state fell through to NORMAL - ordinal 3 and 130 kWh - in training and at
# inference alike. Changing the keys changes two numeric features for those
# states, so the models have to be retrained after this (python src/train.py).
ENERGY_MEAN_BY_STATE = {
    "NEW": 80,
    "EXCELLENT": 90,
    "FULLY RENOVATED": 110,
    "UNDER CONSTRUCTION": 100,
    "NORMAL": 130,
    "TO RENOVATE": 180,
    "TO RESTORE": 220,
    "TO DEMOLISH": 300,
}

STATE_MAPPING = {
    "TO DEMOLISH": 0,
    "TO RESTORE": 1,
    "TO RENOVATE": 2,
    "NORMAL": 3,
    "UNDER CONSTRUCTION": 4,
    "FULLY RENOVATED": 5,
    "EXCELLENT": 6,
    "NEW": 7,
}

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering used during training and prediction.
    Must be identical for train and inference.
    """
    df = df.copy()

    # Ensure required columns exist
    defaults = {
        "property_state": "NORMAL",
        "build_year": CURRENT_YEAR,
        "livable_surface": 0,
        "total_surface": None,
        "garage": 0,
        "terrace": 0,
        "swimming_pool": 0,
        "bedroom_count": 1,
        "energy_consumption_kWh_m2_year": 0,
        "preschool_distance_m": 0,
        "train_station_distance_m": 0,
        "supermarket_distance_m": 0,
    }

    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value

    # Property state cleaning
    # Upper case with single spaces is the canonical spelling (see the two
    # dictionaries above); an underscore or stray whitespace from a caller is
    # folded into it rather than silently becoming NORMAL.
    df["property_state"] = (
        df["property_state"]
        .fillna("NORMAL")
        .astype(str)
        .str.upper()
        .str.replace("_", " ", regex=False)
        .str.split()
        .str.join(" ")
    )

    # Energy consumption logic
    def replace_energy(row):
        energy = row["energy_consumption_kWh_m2_year"]
        if pd.isna(energy) or energy == 0:
            return ENERGY_MEAN_BY_STATE.get(
                row["property_state"],
                ENERGY_MEAN_BY_STATE["NORMAL"]
            )
        return energy

    df["energy_consumption_kWh_m2_year"] = df.apply(replace_energy, axis=1)

    # Total surface
    df["total_surface"] = (
        df["total_surface"]
        .fillna(df["livable_surface"] * 1.2)
    )

    # Property age
    df["build_year"] = df["build_year"].fillna(CURRENT_YEAR)
    df["property_age"] = CURRENT_YEAR - df["build_year"]

    # Swimming pool conversion
    df["swimming_pool"] = df["swimming_pool"].fillna(0).astype(int)

    # Property state encoding
    df["property_state_encoded"] = (
        df["property_state"]
        .map(STATE_MAPPING)
        .fillna(STATE_MAPPING["NORMAL"])
        .astype(int)
    )

    return df


def _key(text) -> str:
    """Comparison key: lowercase, no accents, spaces/underscores/hyphens unified."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[\s_\-]+", "-", text.strip().lower())


# Belgian municipalities have an official name per language, and the training
# data keeps only one of them: mostly the Dutch name in Flanders and Brussels,
# the French name in Wallonia, with a few English exceptions ("Antwerp",
# "Brussels", "Liege"). A user who types the name in the other language got no
# match, and with it the "city not in the training data" note plus a worse
# estimate, for cities the models know perfectly well.
#
# Alias -> the spelling used in models/categories.json. Only genuine
# translations of the same municipality belong here, never a nearby one.
CITY_ALIASES = {
    # Brussels region (training data uses the Dutch commune names)
    "Bruxelles": "Brussels", "Brussel": "Brussels", "Brussel-Stad": "Brussels",
    "Ixelles": "Elsene",
    "Uccle": "Ukkel",
    "Forest": "Vorst",
    "Schaerbeek": "Schaarbeek",
    "Saint-Gilles": "Sint-Gillis",
    "Molenbeek-Saint-Jean": "Sint-Jans-Molenbeek", "Molenbeek": "Sint-Jans-Molenbeek",
    "Saint-Josse-ten-Noode": "Sint-Joost-ten-Node", "Saint-Josse": "Sint-Joost-ten-Node",
    "Woluwe-Saint-Lambert": "Sint-Lambrechts-Woluwe",
    "Woluwe-Saint-Pierre": "Sint-Pieters-Woluwe",
    "Watermael-Boitsfort": "Watermaal-Bosvoorde",
    "Auderghem": "Oudergem",
    "Laeken": "Laken",
    "Sint-Agatha-Berchem": "Berchem-Sainte-Agathe",
    # Flanders (training data uses the Dutch name)
    "Antwerpen": "Antwerp", "Anvers": "Antwerp",
    "Ghent": "Gent", "Gand": "Gent",
    "Bruges": "Brugge",
    "Malines": "Mechelen",
    "Louvain": "Leuven",
    "Courtrai": "Kortrijk",
    "Ostend": "Oostende", "Ostende": "Oostende",
    "Ypres": "Ieper",
    "Saint-Nicolas-Waas": "Sint-Niklaas",
    "Alost": "Aalst",
    "Termonde": "Dendermonde",
    "Tirlemont": "Tienen",
    "Saint-Trond": "Sint-Truiden",
    "Tongres": "Tongeren",
    "Looz": "Borgloon",
    "Roulers": "Roeselare",
    "Furnes": "Veurne",
    "Dixmude": "Diksmuide",
    "Poperingue": "Poperinge",
    "Menin": "Menen",
    "Wervicq": "Wervik",
    "Renaix": "Ronse",
    "Audenarde": "Oudenaarde",
    "Grammont": "Geraardsbergen",
    "Vilvorde": "Vilvoorde",
    "Tervueren": "Tervuren",
    "Lierre": "Lier",
    "Leau": "Zoutleeuw",
    "Hal": "Halle",
    "Bourg-Leopold": "Leopoldsburg",
    "Montaigu": "Scherpenheuvel",
    "Fourons": "Voeren",
    # Wallonia (training data uses the French name)
    "Luik": "Liege",
    "Bergen": "Mons",
    "Namen": "Namur",
    "Doornik": "Tournai",
    "Aarlen": "Arlon",
    "Waver": "Wavre",
    "Nijvel": "Nivelles",
    "Zinnik": "Soignies",
    "Aat": "Ath",
    "Hoei": "Huy",
    "Borgworm": "Waremme",
    "Geldenaken": "Jodoigne",
    "Eigenbrakel": "Braine-l'Alleud",
    "Edingen": "Enghien",
    "Lessen": "Lessines",
    "Moeskroen": "Mouscron",
    "Komen": "Comines",
    "Wezet": "Visé",
    "Tubeke": "Tubize",
    "Genepien": "Genappe",
    "Bitsingen": "Bassenge",
    "s-Gravenbrakel": "Braine-le-Comte",
    "Kasteelbrakel": "Braine-le-Château",
    "Elzele": "Ellezelles",
    "Vloesberg": "Flobecq",
    "Opzullik": "Silly",
    "Herzeeuw": "Herseaux",
}

_CITY_ALIAS_BY_KEY = {_key(alias): canonical for alias, canonical in CITY_ALIASES.items()}


def load_vocabulary(path: Path = VOCABULARY_PATH) -> dict:
    """Load the categories the models were trained on (scripts/export_categories.py)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_training_vocabulary(data: dict, vocabulary: dict) -> tuple[dict, list[str]]:
    """
    Rewrite categorical inputs into the exact strings the models were fitted on.

    The TargetEncoder maps any unseen string to the global mean, so "HOUSE",
    "Liège" or "TO_RENOVATE" made the model ignore property type, province and
    state entirely (a house and an apartment got the same price). This only
    changes spelling, never meaning, and is idempotent, so training-style input
    passes through unchanged. property_state ends up as the uppercase, space
    separated form that add_features() produced at training time.

    Returns (normalized copy of data, notes for the user about the city).
    Raises ValueError for a property type, state or province the models never saw.
    """
    data = dict(data)
    notes = []

    property_type = str(data.get("property_type", "")).strip().lower()
    if property_type not in vocabulary["property_types"]:
        raise ValueError(f"Unknown property_type '{data.get('property_type')}'. "
                         f"Expected one of {vocabulary['property_types']}.")
    data["property_type"] = property_type

    state = re.sub(r"[\s_]+", " ", str(data.get("property_state") or "NORMAL")).strip().upper()
    if state not in vocabulary["property_states"]:
        raise ValueError(f"Unknown property_state '{data.get('property_state')}'. "
                         f"Expected one of {vocabulary['property_states']}.")
    data["property_state"] = state

    provinces = vocabulary["provinces"]
    province_by_key = {}
    for slug, info in provinces.items():
        for name in [slug, info["label"], *info["aliases"]]:
            province_by_key[_key(name)] = slug
    province = province_by_key.get(_key(data.get("province", "")))
    if province is None:
        raise ValueError(f"Unknown province '{data.get('province')}'. "
                         f"Expected one of {[p['label'] for p in provinces.values()]}.")
    data["province"] = province

    # The city encoder is independent of the province, so a known spelling from
    # any province is still worth using; the user's own province is preferred.
    # That ordering also settles the few aliases whose target name exists in two
    # provinces: "Forest" resolves to the Brussels "Vorst" for a Brussels input.
    city_key = _key(data.get("city", ""))
    # A French or Dutch name for the same municipality resolves to the single
    # spelling the models were fitted on (see CITY_ALIASES).
    city_key = _key(_CITY_ALIAS_BY_KEY.get(city_key, city_key))
    candidates = [provinces[province]["cities"]] + [
        info["cities"] for slug, info in provinces.items() if slug != province
    ]
    match = next(((city, cities[city]) for cities in candidates for city in cities
                  if _key(city) == city_key), None)
    if match is None:
        notes.append(f"'{data.get('city')}' does not appear in the training data, so the "
                     "estimate relies on the province and the property itself, not the city.")
    else:
        data["city"], n_rows = match
        if n_rows < LOW_SAMPLE_CITY:
            notes.append(f"Only {n_rows} training listings in {data['city']}, so the city's own "
                         "price level carries little weight in this estimate.")

    return data, notes