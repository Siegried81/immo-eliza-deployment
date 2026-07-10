import random
import sys
import json
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "monitoring" / "logs.json"
from monitoring.monitor import log_prediction

def generate_realistic_data():
    """
    Generates data mimicking the full schema of training_baseline.csv.
    """
    build_year = random.randint(1960, 2026)
    livable_surface = random.gauss(120, 30)
    
    return {
        "property_type": "house",
        "property_id": random.randint(10000, 99999),
        "postcode": random.randint(1000, 9000),
        "city": "Brussels",
        "province": "Brussels",
        "address": "Generic Street",
        "latitude": 50.85,
        "longitude": 4.35,
        "property_state": "good",
        "build_year": build_year,
        "bedroom_count": random.choices([1, 2, 3, 4], weights=[10, 30, 40, 20])[0],
        "livable_surface": livable_surface,
        "total_surface": random.gauss(150, 50),
        "garage": random.choices([0, 1], weights=[70, 30])[0],
        "terrace": random.choices([0, 1], weights=[40, 60])[0],
        "energy_consumption_kWh/m2/year": random.uniform(100, 350),
        "swimming_pool": random.choices([0, 1], weights=[95, 5])[0],
        "preschool_distance_m": random.randint(200, 1500),
        "train_station_distance_m": random.randint(500, 3000),
        "supermarket_distance_m": random.randint(300, 2000),
        "nearest_city": "Brussels",
        "nearest_city_distance_km": random.uniform(0, 20),
        "price_per_m2": random.gauss(2500, 500),
        "energy_consumption_kWh_m2_year": random.uniform(100, 350),
        "property_age": 2026 - build_year,
        "property_state_encoded": random.randint(0, 4)
    }

def populate_logs(n=8000):
    if LOG_FILE.exists():
        LOG_FILE.unlink()
        print("Old logs.json deleted.")

    print(f"Generating {n} log entries...")
    for _ in range(n):
        fake_data = generate_realistic_data()
        fake_prediction = random.gauss(350000, 75000)
        log_prediction(fake_data, fake_prediction)
    print("Done!")

if __name__ == "__main__":
    populate_logs(8000)