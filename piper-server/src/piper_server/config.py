from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    repo_id: str


@dataclass(frozen=True)
class AppConfig:
    cache_dir: Path
    default_model: str
    models: dict[str, ModelConfig]


def load_config(path: str) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text()) or {}
    models_list = data.get("models")
    if not isinstance(models_list, list) or not models_list:
        raise ValueError("Config must define a non-empty 'models' list.")

    models: dict[str, ModelConfig] = {}
    for item in models_list:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be a mapping.")
        name = item.get("name")
        repo_id = item.get("repo_id")
        if not name or not repo_id:
            raise ValueError("Each model entry must include 'name' and 'repo_id'.")
        models[name] = ModelConfig(name=name, repo_id=repo_id)

    default_model = data.get("default_model") or models_list[0]["name"]
    if default_model not in models:
        raise ValueError("default_model must match one of the configured model names.")

    env_cache_dir = os.getenv("PIPER_CACHE_DIR") or os.getenv("HF_HOME")
    cache_dir = Path(env_cache_dir or data.get("cache_dir") or (config_path.parent / "hf-cache")).expanduser()
    return AppConfig(cache_dir=cache_dir, default_model=default_model, models=models)
