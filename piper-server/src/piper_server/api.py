from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import Response

from .config import AppConfig
from .model_manager import PiperModelManager

app_config: AppConfig | None = None
model_manager: PiperModelManager | None = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    if app_config is None:
        raise RuntimeError("Application config has not been initialized.")
    app_config.cache_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="piper-server", version="0.1.0", lifespan=_lifespan)


def configure_app(config: AppConfig) -> None:
    global app_config, model_manager
    app_config = config
    model_manager = PiperModelManager(config)


def _get_manager() -> PiperModelManager:
    if model_manager is None or app_config is None:
        raise RuntimeError("Application is not configured.")
    return model_manager


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/audio/speech")
async def speech(
    request: Request,
    input_text: Optional[str] = Form(None, alias="input"),
    model: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    speed: Optional[float] = Form(None),
) -> Response:
    del voice
    if app_config is None:
        raise HTTPException(status_code=500, detail="application is not configured")

    payload: dict[str, Any] = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()

    input_value = payload.get("input", input_text)
    model_name = payload.get("model", model) or app_config.default_model
    response_format_value = payload.get("response_format", response_format) or "wav"
    speed_value = payload.get("speed", speed)
    if speed_value is None:
        speed_value = 1.0

    if input_value is None:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": "missing",
                    "loc": ["body", "input"],
                    "msg": "Field required",
                    "input": None,
                }
            ],
        )

    if response_format_value not in {"wav", "pcm"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format_value}",
                    "type": "invalid_request_error",
                }
            },
        )

    try:
        if response_format_value == "pcm":
            audio_bytes = _get_manager().synthesize_pcm(
                model_name=model_name,
                text=str(input_value),
                speed=float(speed_value),
            )
            media_type = "audio/pcm"
        else:
            audio_bytes = _get_manager().synthesize_wav(
                model_name=model_name,
                text=str(input_value),
                speed=float(speed_value),
            )
            media_type = "audio/wav"
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        ) from exc

    return Response(content=audio_bytes, media_type=media_type)
