"""
Immo Eliza web UI: collects one property, asks the FastAPI service for a price
and shows the estimate with its likely range, segment and caveats.

Every choice list comes from models/categories.json (the categories the models
were trained on), so a user can only pick values the model actually knows.
Province and city sit outside any st.form on purpose: the city list must update
as soon as the province changes. Only the "Estimate" button calls the API.

Run:
    streamlit run streamlit/app.py      (API_URL defaults to http://localhost:8000)
"""
import os
import sys
from pathlib import Path

import requests
import streamlit as st

# `streamlit run streamlit/app.py` puts only streamlit/ on sys.path (Streamlit
# Cloud starts it this way), so add the repo root to import src.features.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import LOW_SAMPLE_CITY, STATE_MAPPING, load_vocabulary  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000").strip().rstrip("/")
API_DOCS_URL = f"{API_URL}/docs"
# Render's free tier sleeps after inactivity; waking it takes up to about a minute.
PREDICT_TIMEOUT_S = 90
HISTORY_SIZE = 5

VOCABULARY = load_vocabulary()
PROVINCES = VOCABULARY["provinces"]
# Best condition first, using the same order as the model's state encoding.
STATES = sorted(VOCABULARY["property_states"],
                key=lambda s: -STATE_MAPPING.get(s.replace(" ", "_"), STATE_MAPPING["NORMAL"]))

CSS = """
<style>
.price-card { background: #ffffff; border: 1px solid #dfe5e2; border-radius: 14px; padding: 1.2rem 1.4rem; }
.price-label { color: #5b6b66; font-size: .9rem; margin-bottom: .2rem; }
.price-value { font-size: 2.4rem; font-weight: 700; color: #16302b; line-height: 1.1; }
.badge { display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem;
         font-weight: 600; margin-left: .5rem; vertical-align: middle; }
.badge-standard { background: #e3f1ec; color: #1f6f5c; }
.badge-luxury { background: #f6ead7; color: #8a5a12; }
.range { position: relative; height: 10px; background: #e7ece9; border-radius: 6px; margin: 1.6rem 0 .4rem; }
.range-fill { position: absolute; height: 100%; background: #9ccfbf; border-radius: 6px; }
.range-marker { position: absolute; top: -5px; width: 4px; height: 20px; background: #1f6f5c; border-radius: 2px; }
.range-labels { display: flex; justify-content: space-between; color: #5b6b66; font-size: .8rem; }
</style>
"""


def euro(value: float) -> str:
    """Format a price the Belgian way: € 283.578."""
    return "€ " + f"{value:,.0f}".replace(",", ".")


@st.cache_data(ttl=60, show_spinner=False)
def api_is_awake() -> bool:
    """Ping the API at most once a minute; a sleeping Render instance times out."""
    try:
        return requests.get(f"{API_URL}/ping", timeout=3).ok
    except requests.RequestException:
        return False


def request_estimate(payload: dict) -> tuple[dict | None, str | None]:
    """POST the property to /predict. Returns (result, error message)."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=PREDICT_TIMEOUT_S)
    except requests.exceptions.Timeout:
        return None, "The API took too long to answer. It may still be waking up: try again in a moment."
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is the backend running?"

    if response.ok:
        return response.json(), None
    if response.status_code == 422:
        detail = response.json().get("detail")
        if isinstance(detail, list):  # pydantic field errors
            detail = "; ".join(f"{'.'.join(map(str, d['loc'][1:]))}: {d['msg']}" for d in detail)
        return None, f"Some inputs were rejected: {detail}"
    return None, f"The API returned an error ({response.status_code}). Please try again later."


def range_bar(lower: float, prediction: float, upper: float) -> str:
    """HTML bar of the 10th-90th percentile interval with a marker at the estimate."""
    low, high = min(lower, prediction), max(upper, prediction)
    span = (high - low) or 1
    left = (lower - low) / span * 100
    width = (upper - lower) / span * 100
    marker = (prediction - low) / span * 100
    return (f'<div class="range"><div class="range-fill" style="left:{left:.1f}%;width:{width:.1f}%"></div>'
            f'<div class="range-marker" style="left:calc({marker:.1f}% - 2px)"></div></div>'
            f'<div class="range-labels"><span>{euro(lower)}</span><span>{euro(upper)}</span></div>')


def show_result(result: dict, livable_surface: int) -> None:
    """Price card, range, key figures and caveats for one estimate."""
    segment = result["segment"]
    badge = ("<span class='badge badge-luxury'>Luxury model</span>" if segment == "luxury"
             else "<span class='badge badge-standard'>Standard model</span>")
    interval = result.get("prediction_interval")
    range_html = ""
    if interval:
        range_html = (range_bar(interval["lower"], result["prediction"], interval["upper"])
                      + "<div class='price-label' style='margin-top:.3rem'>"
                        "Likely range: calibrated to hold the real price for 8 listings in 10</div>")
    st.markdown(
        f"<div class='price-card'><div class='price-label'>Estimated price {badge}</div>"
        f"<div class='price-value'>{euro(result['prediction'])}</div>{range_html}</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Price per m² (livable)", euro(result["prediction"] / livable_surface))
    col2.metric("Luxury probability", f"{result.get('luxury_probability', 0):.0%}",
                help="Output of the routing classifier. Above the threshold, the luxury model prices the property.")

    if segment == "luxury":
        st.warning("Priced by the luxury model (properties around €1.9M and above). It learned from "
                   "about 125 listings, so expect errors of around a third of the price, and no range "
                   "is available for this segment.")
    for note in result.get("notes", []):
        st.info(note)
    st.caption("Estimation tool, not a valuation. On listings never seen in training, the average error "
               "is 22% below €1M (about €70k) and prices are overestimated by about 7% there.")


st.set_page_config(page_title="Immo Eliza · Price estimator", page_icon="🏠", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("history", [])

with st.sidebar:
    st.subheader("🏠 Immo Eliza")
    st.write("Price estimates for Belgian houses and apartments, from an XGBoost model "
             "trained on ~15,700 listings scraped from Immovlan.")
    if api_is_awake():
        st.success("API online", icon="🟢")
    else:
        st.warning("API asleep or unreachable. The first estimate may take up to a minute.", icon="🟠")
    st.markdown(f"[API documentation]({API_DOCS_URL})")
    st.divider()
    st.caption("How it works: a routing classifier sends likely luxury properties (≈ €1.9M+) to a "
               "dedicated model; two quantile models (10th and 90th percentile) give the range for the others.")

st.title("What is this property worth?")
st.write("Describe the property, then click **Estimate**. Fields marked * are required.")

inputs, output = st.columns([3, 2], gap="large")

with inputs:
    with st.container(border=True):
        st.markdown("**📍 Location**")
        c1, c2, c3 = st.columns([2, 2, 1])
        province = c1.selectbox("Province *", list(PROVINCES), format_func=lambda p: PROVINCES[p]["label"],
                                index=list(PROVINCES).index("brussels"))
        cities = PROVINCES[province]["cities"]
        city_names = sorted(cities)
        most_listed = max(cities, key=cities.get)
        city = c2.selectbox("City *", city_names, index=city_names.index(most_listed),
                            help="Only cities present in the training data are listed.")
        postcode = c3.number_input("Postcode *", min_value=1000, max_value=9999, value=1000, step=1)
        if cities[city] < LOW_SAMPLE_CITY:
            st.caption(f"Only {cities[city]} listings in {city}: the estimate leans on the province.")

    with st.container(border=True):
        st.markdown("**🏡 Property**")
        c1, c2 = st.columns(2)
        property_type = c1.radio("Type *", VOCABULARY["property_types"], horizontal=True,
                                 format_func=str.title, index=VOCABULARY["property_types"].index("house"))
        property_state = c2.selectbox("Condition *", STATES, index=STATES.index("NORMAL"),
                                      format_func=str.capitalize)
        c1, c2, c3 = st.columns(3)
        livable_surface = c1.number_input("Livable surface (m²) *", min_value=10, max_value=2000, value=120, step=5)
        total_surface = c2.number_input("Plot surface (m²)", min_value=0, max_value=100_000, value=0, step=10,
                                        help="Leave 0 if unknown or for an apartment: it is then estimated "
                                             "from the livable surface, as during training.")
        bedroom_count = c3.number_input("Bedrooms", min_value=0, max_value=20, value=3)
        c1, c2 = st.columns(2)
        build_year = c1.number_input("Build year", min_value=1800, max_value=2026, value=1975)
        energy_known = c2.toggle("I know the energy consumption", value=False,
                                 help="Otherwise a typical value for the property's condition is used.")
        energy = (c2.number_input("Energy consumption (kWh/m²/year)", min_value=1, max_value=1000, value=250)
                  if energy_known else 0)
        c1, c2, c3 = st.columns(3)
        garage = c1.checkbox("Garage")
        terrace = c2.checkbox("Terrace")
        swimming_pool = c3.checkbox("Swimming pool")

    with st.container(border=True):
        st.markdown("**🚉 Surroundings** (distance in metres)")
        c1, c2, c3 = st.columns(3)
        # Defaults are the training medians, so untouched fields stay neutral.
        train_station = c1.number_input("Train station", min_value=0, max_value=50_000, value=2000, step=100)
        supermarket = c2.number_input("Supermarket", min_value=0, max_value=50_000, value=700, step=50)
        preschool = c3.number_input("Preschool", min_value=0, max_value=50_000, value=500, step=50)

    clicked = st.button("Estimate", type="primary", use_container_width=True)

payload = {
    "postcode": int(postcode), "province": province, "city": city,
    "property_type": property_type, "property_state": property_state,
    "livable_surface": int(livable_surface), "total_surface": int(total_surface) or None,
    "bedroom_count": int(bedroom_count), "build_year": int(build_year),
    "garage": int(garage), "terrace": int(terrace), "swimming_pool": swimming_pool,
    "energy_consumption_kWh_m2_year": int(energy),
    "preschool_distance_m": int(preschool), "train_station_distance_m": int(train_station),
    "supermarket_distance_m": int(supermarket),
}

with output:
    if clicked:
        with st.spinner("Estimating… (the first call can take up to a minute while the API wakes up)"):
            result, error = request_estimate(payload)
        if error:
            st.error(error)
        else:
            st.session_state["last"] = (result, int(livable_surface))
            st.session_state["history"].insert(0, {
                "Type": property_type.title(), "City": city, "m²": int(livable_surface),
                "Condition": property_state.capitalize(), "Estimate": euro(result["prediction"]),
            })
            del st.session_state["history"][HISTORY_SIZE:]

    if "last" in st.session_state:
        show_result(*st.session_state["last"])
    else:
        with st.container(border=True):
            st.markdown("**Your estimate will appear here.**")
            st.caption("You get a price, the range it most likely falls in, the price per m² and the "
                       "caveats that apply to this property.")

    if len(st.session_state["history"]) > 1:
        st.markdown("**Recent estimates**")
        st.dataframe(st.session_state["history"], hide_index=True, use_container_width=True)
