# Piper Server

Standalone OpenAI-compatible text-to-speech server for Piper.

## Install

```bash
cd /Users/drascom/Documents/work/voice-kit/piper-server
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Run

```bash
cd /Users/drascom/Documents/work/voice-kit/piper-server
HF_HOME=/Users/drascom/Documents/work/voice-kit/hf-cache \
uv run piper-server --config config.turkish.yaml --port 9200 --reload
```

## Test

```bash
curl http://localhost:9200/health
```

```bash
curl -X POST "http://localhost:9200/v1/audio/speech" \
  -F "input=Merhaba, bu bir Piper testidir." \
  -F "model=turkish" \
  -F "response_format=wav" \
  --output speech.wav
```

## Notes

- This project is isolated from the main app code in `/Users/drascom/Documents/work/voice-kit`.
- The first run downloads the selected Piper model into the shared cache at `/Users/drascom/Documents/work/voice-kit/hf-cache`.
- In config mode, the request `model` must match a configured `name`.
