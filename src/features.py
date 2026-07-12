import pandas as pd

CURRENT_YEAR = 2026

ENERGY_MEAN_BY_STATE = {
    "NEW": 80,
    "EXCELLENT": 90,
    "FULLY_RENOVATED": 110,
    "UNDER_CONSTRUCTION": 100,
    "NORMAL": 130,
    "TO_RENOVATE": 180,
    "TO_RESTORE": 220,
    "TO_DEMOLISH": 300,
}

STATE_MAPPING = {
    "TO_DEMOLISH": 0,
    "TO_RESTORE": 1,
    "TO_RENOVATE": 2,
    "NORMAL": 3,
    "UNDER_CONSTRUCTION": 4,
    "FULLY_RENOVATED": 5,
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
    df["property_state"] = (
        df["property_state"]
        .fillna("NORMAL")
        .astype(str)
        .str.upper()
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