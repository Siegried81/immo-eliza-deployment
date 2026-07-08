import streamlit as st
import requests
from pathlib import Path
import os


# =====================================================
# CONFIG
# =====================================================

API_URL = str(os.getenv("API_URL", "https://immo-eliza-deployment-ujgj.onrender.com/predict")).strip()

PROPERTY_STATES = [
    "New",
    "Excellent",
    "Fully renovated",
    "Under construction",
    "Normal",
    "To renovate",
    "To restore",
    "To demolish"
]


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Immo Eliza Predictor",
    page_icon="🏠",
    layout="wide" # Changed to wide to better accommodate 3 columns
)


# =====================================================
# HEADER
# =====================================================

st.title("🏠 Immo Eliza Predictor")
st.write("Belgian property price prediction using XGBoost")

# Create 3 columns
col1, col2, col3 = st.columns(3)

# =====================================================
# LOCATION (Column 1)
# =====================================================

with col1:
    st.header("📍 Location")

    postcode = st.number_input(
        "Postal Code",
        min_value=1000,
        max_value=9999,
        value=1000
    )

    province = st.selectbox(
        "Province",
        [
            "Brussels",
            "Antwerp",
            "East Flanders",
            "West Flanders",
            "Flemish Brabant",
            "Walloon Brabant",
            "Hainaut",
            "Liège",
            "Limburg",
            "Luxembourg",
            "Namur"
        ]
    )

    city = st.text_input(
        "City",
        value="Brussels"
    )


# =====================================================
# PROPERTY (Column 2)
# =====================================================

with col2:
    st.header("🏡 Property")

    property_type = st.selectbox(
        "Type",
        [
            "House",
            "Apartment"
        ]
    )

    property_state = st.selectbox(
        "State",
        PROPERTY_STATES,
        index=4
    )

    livable_surface = st.number_input(
        "Livable Surface",
        min_value=1.0,
        value=80.0
    )

    total_surface = st.number_input(
        "Total Surface",
        min_value=0.0,
        value=100.0
    )

    bedroom_count = st.number_input(
        "Bedrooms",
        min_value=0,
        value=2
    )

    build_year = st.number_input(
        "Build Year",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    garage = st.number_input(
        "Garage",
        min_value=0,
        value=0
    )

    garden_m2 = st.number_input(
        "Garden m²",
        min_value=0.0,
        value=0.0
    )

    terrace = st.number_input(
        "Terrace m²",
        min_value=0.0,
        value=0.0
    )

    swimming_pool = st.checkbox(
        "Swimming Pool"
    )


# =====================================================
# DISTANCES & ENERGY (Column 3)
# =====================================================

with col3:
    st.header("🌱 Distances & Energy")

    energy_consumption = st.number_input(
        "Energy consumption",
        min_value=0.0,
        value=0.0,
        help="0 means automatic replacement using the average consumption of the selected property state."
    )

    preschool_distance_m = st.number_input(
        "Preschool distance",
        min_value=0.0,
        value=500.0
    )

    train_station_distance_m = st.number_input(
        "Train station distance",
        min_value=0.0,
        value=800.0
    )

    supermarket_distance_m = st.number_input(
        "Supermarket distance",
        min_value=0.0,
        value=400.0
    )


# =====================================================
# PREDICTION
# =====================================================

st.divider()

# Centering the button using columns
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    if st.button("🔮 Predict price", use_container_width=True):

        payload = {

            # Location
            "postcode": postcode,
            "province": province,
            "city": city,

            # Property
            "property_type": property_type,
            "property_state": property_state,
            "livable_surface": livable_surface,
            "total_surface": total_surface,
            "bedroom_count": bedroom_count,
            "build_year": build_year,
            "garage": garage,
            "garden_m2": garden_m2,
            "terrace": terrace,
            "swimming_pool": swimming_pool,

            # Distances & Energy
            "energy_consumption_kWh_m2_year": energy_consumption,
            "preschool_distance_m": preschool_distance_m,
            "train_station_distance_m": train_station_distance_m,
            "supermarket_distance_m": supermarket_distance_m,

            # Extra fields required by API
            "nearest_city": city,
            "nearest_city_distance_km": 0
        }

        # Clean the URL and inspect its structure
        raw_url = os.getenv("API_URL", "https://immo-eliza-deployment-ujgj.onrender.com/predict")
        API_URL = str(raw_url).strip()

        # Debugging: Use repr() to see hidden characters like \n or spaces
        st.write(f"DEBUG: URL is: '{API_URL}'")
        st.write(f"DEBUG: Representation: {repr(API_URL)}")
                
        try:
            response = requests.post(
                API_URL,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"Estimated price: € {result['prediction']:,.0f}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except Exception as e:
            st.error(f"API connection error: {e}")