import random
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "monitoring" / "logs.json"
from monitoring.monitor import log_prediction

def generate_realistic_data():
    """
    Generates data mimicking a real estate market distribution
    to avoid massive artificial drift.
    """
    return {
        "postcode": random.randint(1000, 9000),
        "build_year": random.randint(1980, 2026),  # More recent buildings
        "bedroom_count": random.choices([1, 2, 3, 4], weights=[10, 30, 40, 20])[0],  # Probabilistic distribution
        "livable_surface": random.gauss(120, 30),  # Mean 120m2, std 30
        "total_surface": random.gauss(150, 50),
        "garage": random.choices([0, 1], weights=[70, 30])[0],  # Fewer houses have garages
        "terrace": random.choices([0, 1], weights=[40, 60])[0],
        "swimming_pool": random.choices([0, 1], weights=[95, 5])[0],  # Pools are rare
        "preschool_distance_m": random.randint(200, 1500),
        "train_station_distance_m": random.randint(500, 3000),
        "supermarket_distance_m": random.randint(300, 2000),
        # NOTE: "nearest_city_distance_km" was intentionally removed here.
        # train.py drops this feature from NUMERIC_FEATURES to reduce drift,
        # so it never appears in training_baseline.csv anyway.
        "energy_consumption_kWh_m2_year": random.uniform(100, 350)  # Narrower range
    }

def populate_logs(n=8000):
    # Clear existing file to ensure a clean slate
    if LOG_FILE.exists():
        LOG_FILE.unlink()
        print("Old logs.json deleted.")

    print(f"Generating {n} realistic log entries...")
    for _ in range(n):
        fake_data = generate_realistic_data()
        # Simulate a prediction
        fake_prediction = random.gauss(350000, 75000)

        log_prediction(fake_data, fake_prediction)
    print(f"Done! {n} entries saved to logs.json.")

if __name__ == "__main__":
    populate_logs(8000)
