"""
Export the categorical vocabulary the trained models know to models/categories.json.

The TargetEncoder inside pipeline.joblib only recognises the exact strings it
was fitted on ("house", "liege", "TO RENOVATE"...). Any other spelling falls
back to the global mean, so the model silently ignores that feature. The API
and the Streamlit UI read this file to translate user input into that
vocabulary and to offer only cities the model has seen.

It replays the same split as src/train.py (TEST_SIZE, SPLIT_SEED) so
the city list and counts match the rows the encoder was actually fitted on.
data/ is not versioned, which is why the result is committed under models/.

Run after every retraining:
    python scripts/export_categories.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from features import add_features  # noqa: E402
from train import SPLIT_SEED, TEST_SIZE  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "clean" / "cleaned_data.json"
OUTPUT_PATH = BASE_DIR / "models" / "categories.json"

# Display label and accepted alternative spellings (English, French, Dutch)
# for each province slug used in the training data.
PROVINCES = {
    "antwerp": ("Antwerp", ["Antwerpen", "Anvers"]),
    "brabant-wallon": ("Walloon Brabant", ["Brabant wallon", "Waals-Brabant"]),
    "brussels": ("Brussels", ["Bruxelles", "Brussel"]),
    "east-flanders": ("East Flanders", ["Flandre orientale", "Oost-Vlaanderen"]),
    "hainaut": ("Hainaut", ["Henegouwen"]),
    "liege": ("Liège", ["Liege", "Luik"]),
    "limburg": ("Limburg", ["Limbourg"]),
    "luxembourg": ("Luxembourg", ["Luxemburg"]),
    "namur": ("Namur", ["Namen"]),
    "vlaams-brabant": ("Flemish Brabant", ["Brabant flamand", "Vlaams-Brabant"]),
    "west-flanders": ("West Flanders", ["Flandre occidentale", "West-Vlaanderen"]),
}


def build_vocabulary(df: pd.DataFrame) -> dict:
    """Return the categories seen in the training split, with city row counts."""
    X = add_features(df).drop(columns=["price"])
    X_train, _ = train_test_split(X, test_size=TEST_SIZE, random_state=SPLIT_SEED)

    unknown = set(X_train["province"].unique()) - set(PROVINCES)
    if unknown:
        raise ValueError(f"Provinces without a label in PROVINCES: {sorted(unknown)}")

    provinces = {}
    for slug, (label, aliases) in PROVINCES.items():
        counts = X_train.loc[X_train["province"] == slug, "city"].value_counts()
        provinces[slug] = {
            "label": label,
            "aliases": aliases,
            "cities": {city: int(n) for city, n in sorted(counts.items())},
        }

    return {
        "property_types": sorted(X_train["property_type"].unique()),
        "property_states": sorted(X_train["property_state"].unique()),
        "provinces": provinces,
    }


if __name__ == "__main__":
    vocabulary = build_vocabulary(pd.read_json(DATA_PATH))
    OUTPUT_PATH.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=1), encoding="utf-8")
    n_cities = sum(len(p["cities"]) for p in vocabulary["provinces"].values())
    print(f"Wrote {OUTPUT_PATH} ({len(vocabulary['provinces'])} provinces, {n_cities} cities)")
