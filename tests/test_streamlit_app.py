"""
End-to-end checks of the Streamlit UI with Streamlit's AppTest.

The API is mocked by replacing requests.get/post, so these tests check what a
user sees and what the UI sends, without any network call.
"""
import sys
import time
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

import streamlit as st

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit" / "app.py")

STANDARD = {"prediction": 283578.0, "prediction_interval": {"lower": 240000.0, "upper": 330000.0},
            "likely_range": {"lower": 250000.0, "upper": 320000.0},
            "currency": "EUR", "status": "success", "segment": "standard",
            "luxury_probability": 0.02, "notes": []}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code, self._body = status_code, body
        self.ok = status_code < 400

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
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
    assert [m.label for m in at.metric] == ["Price per m² (livable)", "Chance of €1.95M+"]
    assert ('aria-label="Estimate € 283.578, wider range from € 240.000 to € 330.000, '
            'most likely from € 250.000 to € 320.000"') in html_of(at)
    assert "Wider range (-15% / +16%)" in html_of(at)
    assert "Most likely</b> € 250.000 to € 320.000 (-12% / +13%)" in html_of(at)
    assert 'class="range-likely"' in html_of(at)


def test_an_older_api_without_the_likely_band_still_shows_the_range(api):
    api.reply = FakeResponse(200, {k: v for k, v in STANDARD.items() if k != "likely_range"})
    at = run_app()
    at.button[0].click().run()
    assert "Wider range" in html_of(at)
    assert "Most likely" not in html_of(at)
    assert any("half of the estimates" in c.value for c in at.caption)


def test_an_edited_input_marks_the_estimate_as_stale(api):
    at = run_app()
    at.button[0].click().run()
    assert not at.warning
    at.number_input[1].set_value(300).run()  # livable surface
    assert any("inputs changed" in w.value for w in at.warning)
    at.button[0].click().run()
    assert not at.warning


def test_a_waking_api_is_retried_once(api, monkeypatch):
    replies = [FakeResponse(503, ValueError("html page")), FakeResponse(200, STANDARD)]
    waits = []
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: replies.pop(0))
    monkeypatch.setattr(time, "sleep", waits.append)
    at = run_app()
    at.button[0].click().run()
    assert waits.count(5) == 1 and not at.error  # Streamlit sleeps briefly on its own too
    assert "€ 283.578" in html_of(at)


def test_a_reply_that_is_not_json_is_an_error_message(api):
    api.reply = FakeResponse(200, ValueError("no json"))
    at = run_app()
    at.button[0].click().run()
    assert "not an estimate" in at.error[0].value


def test_a_broken_connection_is_an_error_message(api, monkeypatch):
    def cut(*args, **kwargs):
        raise requests.exceptions.ChunkedEncodingError()
    monkeypatch.setattr(requests, "post", cut)
    at = run_app()
    at.button[0].click().run()
    assert "broke off" in at.error[0].value


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
    # The accuracy caption describes the standard model below €1M only.
    assert not any("half of the estimates" in c.value for c in at.caption)


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


def test_docs_link_follows_the_api_url(api, monkeypatch):
    monkeypatch.setenv("API_URL", "https://api.example.org/")
    monkeypatch.delenv("API_DOCS_URL", raising=False)
    at = run_app()
    assert "[API documentation](https://api.example.org/docs)" in [m.value for m in at.markdown]


def test_empty_docs_url_hides_the_link(api, monkeypatch):
    """An API that only listens on the UI's localhost sets API_DOCS_URL to ""."""
    monkeypatch.setenv("API_DOCS_URL", "")
    at = run_app()
    assert not any("API documentation" in m.value for m in at.markdown)


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
