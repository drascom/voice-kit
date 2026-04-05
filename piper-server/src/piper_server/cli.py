from __future__ import annotations

import argparse
import os

import uvicorn

from .api import app, configure_app
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Piper TTS API server.")
    parser.add_argument("--config", "-f", required=True, help="Path to YAML config.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=9200, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    args = parser.parse_args()

    config = load_config(args.config)
    os.environ.setdefault("HF_HOME", str(config.cache_dir))
    configure_app(config)

    app_target = "piper_server.api:app" if args.reload else app
    uvicorn.run(app_target, host=args.host, port=args.port, reload=args.reload)
