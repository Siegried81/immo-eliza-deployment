"""
End-to-end checks of the Streamlit UI with Streamlit's AppTest.

The API is mocked by replacing requests.get/post, so these tests check what a
user sees and what the UI sends, without any network call.
"""
import sys
from pathlib import Path

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit" / "app.py")

STANDARD = {"prediction": 283578.0, "prediction_interval": {"lower": 240000.0, "upper": 330000.0},
            "currency": "EUR", "status": "success", "segment": "standard",
            "luxury_probability": 0.02, "notes": []}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code, self._body = status_code, body
        self.ok = status_code < 400

    def json(self):
        return self._body


@pytest.fixture
def api(monkeypatch):
    """Fake API: records every /predict payload and answers with `api.reply`."""
    class Api:
        reply = FakeResponse(200, STANDARD)
        sent = []

    def fake_post(url, json, timeout):
        Api.sent.append(json)
        return Api.reply

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", lambda url, timeout: FakeResponse(200, {"status": "alive"}))
    st.cache_data.clear()
    return Api


def run_app():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def html_of(at):
    return " ".join(m.value for m in at.markdown)


def test_first_load_shows_placeholder_and_calls_no_prediction(api):
    at = run_app()
    assert "Your estimate will appear here" in html_of(at)
    assert api.sent == []


def test_estimate_sends_training_vocabulary_and_shows_price(api):
    at = run_app()
    at.button[0].click().run()

    payload = api.sent[-1]
    assert payload["province"] == "brussels"
    assert payload["property_type"] == "house"
    assert payload["property_state"] == "NORMAL"
    assert payload["total_surface"] is None  # 0 in the UI means unknown
    assert "€ 283.578" in html_of(at)
    assert "range" in html_of(at)
    assert [m.label for m in at.metric] == ["Price per m² (livable)", "Luxury probability"]


def test_changing_province_updates_the_city_list(api):
    at = run_app()
    at.selectbox[0].select("liege").run()
    assert "Liege" in at.selectbox[1].options
    assert "Anderlecht" not in at.selectbox[1].options


def test_luxury_result_has_warning_and_no_range(api):
    api.reply = FakeResponse(200, {**STANDARD, "prediction": 3_200_000.0, "segment": "luxury",
                                   "prediction_interval": None, "luxury_probability": 0.91})
    at = run_app()
    at.button[0].click().run()
    assert "Luxury model" in html_of(at)
    assert 'class="range"' not in html_of(at)
    assert any("luxury model" in w.value for w in at.warning)


def test_rejected_input_shows_api_detail(api):
    api.reply = FakeResponse(422, {"detail": "Unknown province 'Utrecht'."})
    at = run_app()
    at.button[0].click().run()
    assert "Unknown province 'Utrecht'" in at.error[0].value


def test_unreachable_api_shows_connection_error(api, monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(requests, "post", refuse)
    at = run_app()
    at.button[0].click().run()
    assert "Cannot reach the API" in at.error[0].value


def test_app_starts_without_repo_root_on_path(tmp_path):
    """`streamlit run streamlit/app.py` only puts streamlit/ on sys.path, so the
    app must find src/ by itself. Run in a fresh interpreter outside the repo."""
    import os
    import subprocess

    script = ("from streamlit.testing.v1 import AppTest\n"
              f"at = AppTest.from_file({APP_PATH!r}, default_timeout=30)\n"
              "at.run()\n"
              "print('EXC', [e.value for e in at.exception])\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["API_URL"] = "http://127.0.0.1:9"  # nothing listens: the ping fails fast
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert "EXC []" in result.stdout, result.stdout + result.stderr
