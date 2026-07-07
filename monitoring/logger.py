import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("monitoring/logs.json")


def log_prediction(input_data, prediction):
    LOG_FILE.parent.mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction
    }

    if LOG_FILE.exists():
        logs = json.loads(LOG_FILE.read_text())
    else:
        logs = []

    logs.append(entry)

    LOG_FILE.write_text(json.dumps(logs, indent=2))