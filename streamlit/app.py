import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").strip().rstrip("/")

st.set_page_config(page_title="Immo Eliza Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Immo Eliza Predictor")

# Form inputs
col1, col2, col3 = st.columns(3)
with col1:
    postcode = st.number_input("Postal Code", min_value=1000, max_value=9999, value=1000)
    province = st.selectbox("Province", ["Brussels", "Antwerp", "East Flanders", "West Flanders", "Flemish Brabant", "Walloon Brabant", "Hainaut", "Liège", "Limburg", "Luxembourg", "Namur"])
    city = st.text_input("City", value="Brussels")

with col2:
    property_type = st.selectbox("Type", ["HOUSE", "APARTMENT"])
    property_state = st.selectbox("State", ["NEW", "EXCELLENT", "FULLY_RENOVATED", "NORMAL", "TO_RENOVATE"])
    livable_surface = st.number_input("Livable Surface (m²)", min_value=1, value=80)
    total_surface = st.number_input("Total Surface (m²)", min_value=0, value=100)
    bedroom_count = st.number_input("Bedrooms", min_value=0, value=2)
    garage = st.number_input("Garage Spaces", min_value=0, value=0)

with col3:
    energy_consumption = st.number_input("Energy Consumption", min_value=0, value=0)
    swimming_pool = st.checkbox("Swimming Pool")

if st.button("🔮 Predict Price"):
    payload = {
        "postcode": int(postcode),
        "province": province,
        "city": city,
        "property_type": property_type,
        "property_state": property_state,
        "livable_surface": int(livable_surface),
        "total_surface": int(total_surface),
        "bedroom_count": int(bedroom_count),
        "build_year": 2000,
        "garage": int(garage),
        "garden_m2": 0,
        "terrace": 0,
        "swimming_pool": bool(swimming_pool),
        "energy_consumption_kWh_m2_year": int(energy_consumption),
        "preschool_distance_m": 500,
        "train_station_distance_m": 800,
        "supermarket_distance_m": 400
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prediction generated!")
            st.metric("Estimated Price", f"€ {result['prediction']:,.0f}")
            if result.get("segment") == "luxury":
                st.info("🏛️ Routed to luxury-segment model.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"❌ Connection error: {e}")