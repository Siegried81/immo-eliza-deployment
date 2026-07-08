import os
import requests
import streamlit as st

# Application configuration.
# This section defines:
# - The API endpoint used for predictions.
# - The possible values accepted by the machine learning model.
# - The display translations used in the user interface.

API_URL = os.getenv(
    "API_URL",
    "https://immo-eliza-deployment-ujgj.onrender.com/predict"
).strip()

PROPERTY_TYPES = [
    "HOUSE",
    "APARTMENT"
]

PROPERTY_STATES = [
    "NEW",
    "EXCELLENT",
    "FULLY_RENOVATED",
    "UNDER_CONSTRUCTION",
    "NORMAL",
    "TO_RENOVATE",
    "TO_RESTORE",
    "TO_DEMOLISH"
]

PROVINCES = [
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

# Display translations for Dutch-speaking users.
# The original English values are kept for model compatibility.

DUTCH_PROVINCES = {
    "Brussels": "Brussel",
    "Antwerp": "Antwerpen",
    "East Flanders": "Oost-Vlaanderen",
    "West Flanders": "West-Vlaanderen",
    "Flemish Brabant": "Vlaams-Brabant",
    "Walloon Brabant": "Waals-Brabant",
    "Hainaut": "Henegouwen",
    "Liège": "Luik",
    "Limburg": "Limburg",
    "Luxembourg": "Luxemburg",
    "Namur": "Namen",
}

# Streamlit page configuration.

st.set_page_config(
    page_title="Immo Eliza Predictor",
    page_icon="🏠",
    layout="wide"
)

# Application header.

st.title("🏠 Immo Eliza Predictor")
st.write("Belgian property price prediction using XGBoost")

# Create three columns to organize the prediction form.

col1, col2, col3 = st.columns(3)

# Location inputs.
# These values are used as model features.

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
        PROVINCES,
        format_func=lambda p: DUTCH_PROVINCES[p]
    )

    city = st.text_input(
        "City",
        value="Brussels"
    )

# Property characteristics.
# These inputs describe the main attributes of the property.

with col2:
    st.header("🏡 Property")

    property_type = st.selectbox(
        "Type",
        PROPERTY_TYPES,
        format_func=lambda x: x.title()
    )

    property_state = st.selectbox(
        "State",
        PROPERTY_STATES,
        index=4,
        format_func=lambda x: x.replace("_", " ").title()
    )

    livable_surface = st.number_input(
        "Livable Surface (m²)",
        min_value=0,
        value=80,
        step=1,
        format="%d"
    )

    total_surface = st.number_input(
        "Total Surface (m²)",
        min_value=0,
        value=100
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
        "Garage Spaces",
        min_value=0,
        value=0
    )

    garden_m2 = st.number_input(
        "Garden (m²)",
        min_value=0,
        value=0
    )

    terrace = st.number_input(
        "Terrace (m²)",
        min_value=0,
        value=0
    )

    swimming_pool = st.checkbox(
        "Swimming Pool"
    )

# Distance and energy inputs.
# These features provide additional information used by the prediction model.

with col3:
    st.header("🌱 Distances & Energy")

    energy_consumption = st.number_input(
        "Energy Consumption (kWh/m²/year)",
        min_value=0,
        value=0,
        help="0 means automatic replacement using the average consumption of the selected property state."
    )

    preschool_distance_m = st.number_input(
        "Preschool Distance (m)",
        min_value=0,
        value=500
    )

    train_station_distance_m = st.number_input(
        "Train Station Distance (m)",
        min_value=0,
        value=800
    )

    supermarket_distance_m = st.number_input(
        "Supermarket Distance (m)",
        min_value=0,
        value=400
    )

# Prediction request.
# The user inputs are converted into a JSON payload and sent to the API.

st.divider()

left, center, right = st.columns([1, 2, 1])

with center:

    if st.button(
        "🔮 Predict Price",
        use_container_width=True
    ):

        payload = {
            "postcode": postcode,
            "province": province,
            "city": city,
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
            "energy_consumption_kWh_m2_year": energy_consumption,
            "preschool_distance_m": preschool_distance_m,
            "train_station_distance_m": train_station_distance_m,
            "supermarket_distance_m": supermarket_distance_m,
            "nearest_city": city,
            "nearest_city_distance_km": 0
        }

        # Check that the API URL is valid.
        if not API_URL.startswith("http"):

            st.error(
                f"Invalid API URL: '{API_URL}'. Please check your environment variables."
            )

        else:

            try:

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=10
                )

                # Display the prediction returned by the API.
                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        f"Estimated price: € {result['prediction']:,.0f}"
                    )

                # Display API errors returned by the server.
                else:

                    st.error(
                        f"API Error ({response.status_code}): {response.text}"
                    )

            # Handle API connection problems.
            except requests.exceptions.ConnectionError:

                st.error(
                    "Connection error: Unable to reach the API. Please verify that the API service is running."
                )

            # Handle API timeout errors.
            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. Please try again."
                )

            # Handle unexpected errors.
            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )