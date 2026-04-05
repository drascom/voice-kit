from __future__ import annotations

from pathlib import Path

import yaml
from faster_whisper import WhisperModel


def main() -> None:
    config_path = Path("/app/config.turkish.yaml")
    data = yaml.safe_load(config_path.read_text()) or {}
    models = data.get("models") or []

    seen_paths: set[str] = set()
    for item in models:
        model_path = item.get("path")
        if not model_path or model_path in seen_paths:
            continue
        seen_paths.add(model_path)
        model = WhisperModel(model_path)
        del model


if __name__ == "__main__":
    main()
