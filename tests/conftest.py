"""
Shared pytest setup.

Puts the repo root on sys.path and stops the API from appending test requests
to monitoring/logs.json, which would otherwise pollute the drift report.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def no_prediction_logging(monkeypatch):
    import api.app
    monkeypatch.setattr(api.app, "log_prediction", lambda *args, **kwargs: None)
