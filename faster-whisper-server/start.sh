#!/bin/sh
set -eu

python /app/preload_models.py
exec faster-whisper-server --config /app/config.turkish.yaml --port 9100 --host 0.0.0.0
