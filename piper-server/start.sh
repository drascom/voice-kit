#!/bin/sh
set -eu

python /app/preload_models.py
exec piper-server --config /app/config.yaml --port 9200 --host 0.0.0.0
