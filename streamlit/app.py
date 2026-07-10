import os
import requests
import streamlit as st

# Configuration of the API URL 
API_URL = os.getenv("API_URL", "http://localhost:8000").strip()

# Page configuration
st.set_page_config(page_title="Immo Eliza Predictor", page_icon="🏠", layout="wide")

st.title("🏠 Immo Eliza Predictor")
st.write("Belgian property price prediction using XGBoost")

st.markdown("**Note:** Fields marked with an asterisk (*) are mandatory.")

# Configuration lists
PROPERTY_TYPES = ["HOUSE", "APARTMENT"]
PROPERTY_STATES = [
    "NEW", "EXCELLENT", "FULLY_RENOVATED", "UNDER_CONSTRUCTION",
    "NORMAL", "TO_RENOVATE", "TO_RESTORE", "TO_DEMOLISH"
]
PROVINCES = [
    "Brussels", "Antwerp", "East Flanders", "West Flanders",
    "Flemish Brabant", "Walloon Brabant", "Hainaut", "Liège",
    "Limburg", "Luxembourg", "Namur"
]
DUTCH_PROVINCES = {
    "Brussels": "Brussel", "Antwerp": "Antwerpen", "East Flanders": "Oost-Vlaanderen",
    "West Flanders": "West-Vlaanderen", "Flemish Brabant": "Vlaams-Brabant",
    "Walloon Brabant": "Waals-Brabant", "Hainaut": "Henegouwen", "Liège": "Luik",
    "Limburg": "Limburg", "Luxembourg": "Luxemburg", "Namur": "Namen",
}

# User Interface
col1, col2, col3 = st.columns(3)

with col1:
    st.header("📍 Location")
    postcode = st.number_input("Postal Code *", min_value=1000, max_value=9999, value=1000)
    province = st.selectbox("Province *", PROVINCES, format_func=lambda p: DUTCH_PROVINCES[p])
    city = st.text_input("City *", value="Brussels")

with col2:
    st.header("🏡 Property")
    property_type = st.selectbox("Type *", PROPERTY_TYPES, format_func=lambda x: x.title())
    property_state = st.selectbox("State *", PROPERTY_STATES, index=4, format_func=lambda x: x.replace("_", " ").title())
    livable_surface = st.number_input("Livable Surface (m²) *", min_value=1, value=80, step=1, format="%d")
    total_surface = st.number_input("Total Surface (m²)", min_value=0, value=100)
    bedroom_count = st.number_input("Bedrooms", min_value=0, value=2)
    build_year = st.number_input("Build Year", min_value=1800, max_value=2026, value=2000)
    garage = st.number_input("Garage Spaces", min_value=0, value=0)
    garden_m2 = st.number_input("Garden (m²)", min_value=0, value=0)
    terrace = st.number_input("Terrace (m²)", min_value=0, value=0)
    swimming_pool = st.checkbox("Swimming Pool")

with col3:
    st.header("🌱 Distances & Energy")
    energy_consumption = st.number_input("Energy Consumption (kWh/m²/year)", min_value=0, value=0)
    preschool_distance_m = st.number_input("Preschool Distance (m)", min_value=0, value=500)
    train_station_distance_m = st.number_input("Train Station Distance (m)", min_value=0, value=800)
    supermarket_distance_m = st.number_input("Supermarket Distance (m)", min_value=0, value=400)

st.divider()

# Prediction logic
left, center, right = st.columns([1, 2, 1])

with center:
    if st.button("🔮 Predict Price", use_container_width=True):
        payload = {
            "postcode": postcode, "province": province, "city": city,
            "property_type": property_type, "property_state": property_state,
            "livable_surface": livable_surface, "total_surface": total_surface,
            "bedroom_count": bedroom_count, "build_year": build_year,
            "garage": garage, "garden_m2": garden_m2, "terrace": terrace,
            "swimming_pool": swimming_pool,
            "energy_consumption_kWh_m2_year": energy_consumption,
            "preschool_distance_m": preschool_distance_m,
            "train_station_distance_m": train_station_distance_m,
            "supermarket_distance_m": supermarket_distance_m
        }

        try:
            with st.spinner("Calculating prediction..."):
                # Concatenate /predict to use the dynamic base URL[cite: 1]
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=80)

            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Prediction generated successfully!")
                st.metric("Estimated Price", f"€ {result['prediction']:,.0f}")

                if result.get("segment") == "luxury":
                    st.caption("🏛️ This property was routed to the luxury-segment model, trained on a smaller pool of high-end listings.")

                st.caption(
                    "ℹ️ This is a model-generated estimate, not a formal valuation. "
                    "It can differ from the actual market value, especially for atypical "
                    "or very high-end properties, and has historically run slightly above "
                    "eventual sale prices on average."
                )
            elif response.status_code == 400:
                error_data = response.json()
                st.error(f"⚠️ Input Error: {error_data.get('detail', 'Please check your inputs.')}")
            else:
                st.error(f"Server Error ({response.status_code}): Please try again later.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Connection error: Unable to reach the API. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
