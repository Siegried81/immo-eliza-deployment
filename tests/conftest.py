"""
Shared pytest setup.

Stops the API from appending test requests to the prediction log, which
would otherwise pollute the drift report.
"""
import pytest


@pytest.fixture(autouse=True)
def no_prediction_logging(monkeypatch):
    import api.app
    monkeypatch.setattr(api.app, "log_prediction", lambda *args, **kwargs: None)
