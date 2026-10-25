"""
scripts/publish_hf.py: what each Hub repo receives, and that a dry run sends nothing.

The Hub client is replaced by a recorder, so no test makes a network call.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import publish_hf  # noqa: E402

RUNTIME_MODELS = {"pipeline.joblib", "pipeline_luxury.joblib", "luxury_classifier.joblib",
                  "pipeline_lower.joblib", "pipeline_upper.joblib", "luxury_threshold.json",
                  "interval_calibration.json", "categories.json"}


class RecordingApi:
    """Stands in for HfApi: keeps the calls, answers like the Hub would."""
    def __init__(self):
        self.repos, self.commits = [], []

    def create_repo(self, repo_id, **kwargs):
        self.repos.append((repo_id, kwargs))

    def create_commit(self, repo_id, **kwargs):
        self.commits.append((repo_id, kwargs))
        return type("Commit", (), {"commit_url": f"https://huggingface.co/{repo_id}/commit/abc"})()


def test_space_gets_the_runtime_and_the_hub_files_at_its_root():
    files = publish_hf.plan("space")
    assert {"Dockerfile", "start.sh", "README.md", "api/app.py", "streamlit/app.py",
            "src/features.py", "monitoring/monitor.py", "requirements.txt"} <= set(files)
    assert {f"models/{name}" for name in RUNTIME_MODELS} <= set(files)
    assert all(src.exists() for src in files.values())


def test_nothing_untracked_or_test_only_is_published():
    for target in publish_hf.TARGETS:
        for dest in publish_hf.plan(target):
            assert not dest.startswith(("tests/", "data/", ".venv", "notebooks/")), dest
            assert dest != "models/best_XGBoost.json"  # local, never committed


def test_model_repo_holds_the_artifacts_at_its_root_with_the_card():
    files = publish_hf.plan("model")
    assert set(files) == RUNTIME_MODELS | {"README.md"}
    assert files["README.md"].name == "model_card.md"


def test_space_card_declares_docker_on_the_port_the_container_serves():
    card = (publish_hf.HF_DIR / "space" / "README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in card and "app_port: 7860" in card
    assert "--server.port 7860" in (publish_hf.HF_DIR / "space" / "start.sh").read_text()


def test_start_script_has_unix_line_endings():
    """A CRLF start.sh fails in the Linux container with a confusing error."""
    assert b"\r\n" not in (publish_hf.HF_DIR / "space" / "start.sh").read_bytes()


def test_dry_run_uploads_nothing(monkeypatch, capsys):
    def no_push(*args, **kwargs):
        raise AssertionError("a dry run must not push")
    monkeypatch.setattr(publish_hf, "push", no_push)
    assert publish_hf.main(["model"]) == 0
    assert "Dry run: 9 files" in capsys.readouterr().out


@pytest.mark.parametrize("target,repo_type", [("space", "space"), ("model", "model")])
def test_push_creates_the_repo_and_sends_one_commit(target, repo_type):
    api = RecordingApi()
    files = publish_hf.plan(target)
    url = publish_hf.push(target, files, api=api)

    repo_id, kwargs = api.repos[0]
    assert repo_id == publish_hf.TARGETS[target]["repo_id"]
    assert kwargs["repo_type"] == repo_type and kwargs["exist_ok"] is True
    assert kwargs.get("space_sdk") == ("docker" if target == "space" else None)
    _, commit = api.commits[0]
    assert sorted(op.path_in_repo for op in commit["operations"]) == sorted(files)
    assert url.startswith(f"https://huggingface.co/{repo_id}/")
