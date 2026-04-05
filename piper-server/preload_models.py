from __future__ import annotations

from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main() -> None:
    config_path = Path("/app/config.turkish.yaml")
    data = yaml.safe_load(config_path.read_text()) or {}
    models = data.get("models") or []

    seen_repos: set[str] = set()
    for item in models:
        repo_id = item.get("repo_id")
        if not repo_id or repo_id in seen_repos:
            continue
        seen_repos.add(repo_id)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["model.onnx", "config.json", "README.md"],
        )


if __name__ == "__main__":
    main()
