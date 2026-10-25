"""
Publish the app as a Hugging Face Space, or the models as a Hub model repo.

    python scripts/publish_hf.py space            # list what would be uploaded
    python scripts/publish_hf.py model --push     # create the repo and upload

Only files tracked by git are published, plus the Hub-specific files in hf/
(mapped to the repo root, where the Hub expects a README and a Dockerfile).
So a local dataset, a virtualenv or an untracked model can never leak, and what
the Space runs is what the repository says. Dry run by default: publishing is
public, so it takes an explicit --push. Authentication is the token saved by
`hf auth login`; this script never reads or prints it.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
HF_DIR = BASE_DIR / "hf"
USER = "Siegried81"

# git pathspecs of what each repo needs at runtime: no tests, notebooks or data.
TARGETS = {
    "space": {
        "repo_id": f"{USER}/immo-eliza",
        "repo_type": "space",
        "tracked": ["requirements.txt", ".streamlit/config.toml", "api/*.py", "src/*.py",
                    "monitoring/*.py", "streamlit/app.py", "models/*"],
        "extra": {"Dockerfile": HF_DIR / "space" / "Dockerfile",
                  "start.sh": HF_DIR / "space" / "start.sh",
                  "README.md": HF_DIR / "space" / "README.md"},
    },
    "model": {
        "repo_id": f"{USER}/immo-eliza-price-models",
        "repo_type": "model",
        "tracked": ["models/*"],
        "extra": {"README.md": HF_DIR / "model_card.md"},
        # The model repo holds the artifacts at its root, as `models/` does here.
        "strip_prefix": "models/",
    },
}


def tracked_files(pathspecs: list[str]) -> list[str]:
    """Repo-relative paths git tracks under `pathspecs`, in git's order."""
    out = subprocess.run(["git", "ls-files", "--", *pathspecs], cwd=BASE_DIR,
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def plan(target: str) -> dict[str, Path]:
    """Map each path in the Hub repo to the local file uploaded there."""
    spec = TARGETS[target]
    prefix = spec.get("strip_prefix", "")
    files = {path.removeprefix(prefix): BASE_DIR / path for path in tracked_files(spec["tracked"])}
    files.update(spec["extra"])
    return files


def push(target: str, files: dict[str, Path], api=None) -> str:
    """Create the repo if needed and upload every file in one commit; return its URL."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = api or HfApi()
    spec = TARGETS[target]
    kwargs = {"space_sdk": "docker"} if spec["repo_type"] == "space" else {}
    api.create_repo(spec["repo_id"], repo_type=spec["repo_type"], exist_ok=True, **kwargs)
    operations = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src))
                  for dest, src in files.items()]
    commit = api.create_commit(spec["repo_id"], repo_type=spec["repo_type"], operations=operations,
                               commit_message="Publish from immo-eliza-deployment")
    return commit.commit_url


def main(argv=None) -> int:
    """List the upload plan, and run it only with --push."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("target", choices=sorted(TARGETS))
    parser.add_argument("--push", action="store_true", help="upload (default: dry run)")
    args = parser.parse_args(argv)

    files = plan(args.target)
    for dest, src in files.items():
        print(f"{dest:40} <- {src.relative_to(BASE_DIR)}")
    if not args.push:
        print(f"\nDry run: {len(files)} files for {TARGETS[args.target]['repo_id']}. Add --push to upload.")
        return 0
    print(push(args.target, files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
