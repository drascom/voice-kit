#!/bin/sh
set -eu

CONFIG_PATH="${FWS_CONFIG_PATH:-/app/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" /app/preload_models.py
exec faster-whisper-server --config "$CONFIG_PATH" --port 9100 --host 0.0.0.0
