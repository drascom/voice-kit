from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from onnxruntime import InferenceSession
from piper.config import PiperConfig, SynthesisConfig
from piper.voice import PiperVoice

from .config import AppConfig


class PiperModelManager:
    def __init__(self, app_config: AppConfig) -> None:
        self.app_config = app_config
        self._voices: dict[str, PiperVoice] = {}

    def _resolve_model_files(self, model_name: str) -> tuple[Path, Path]:
        model_config = self.app_config.models.get(model_name)
        if model_config is None:
            raise ValueError(f"unknown model: {model_name}")

        os.environ.setdefault("HF_HOME", str(self.app_config.cache_dir))
        repo_path = Path(
            snapshot_download(
                repo_id=model_config.repo_id,
                repo_type="model",
                allow_patterns=["model.onnx", "config.json", "README.md"],
            )
        )
        model_path = repo_path / "model.onnx"
        config_path = repo_path / "config.json"
        if not model_path.exists() or not config_path.exists():
            raise ValueError(f"missing Piper files for model: {model_name}")
        return model_path, config_path

    def load_voice(self, model_name: str) -> PiperVoice:
        cached = self._voices.get(model_name)
        if cached is not None:
            return cached

        model_path, config_path = self._resolve_model_files(model_name)
        session = InferenceSession(str(model_path))
        config = PiperConfig.from_dict(json.loads(config_path.read_text()))
        voice = PiperVoice(session=session, config=config)
        self._voices[model_name] = voice
        return voice

    def synthesize_wav(self, model_name: str, text: str, speed: float) -> bytes:
        if not text.strip():
            raise ValueError("input text is required")
        if speed < 0.25 or speed > 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")

        voice = self.load_voice(model_name)
        audio_chunks: list[np.ndarray] = []
        for chunk in voice.synthesize(
            text,
            SynthesisConfig(length_scale=1.0 / speed),
        ):
            audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
            audio_chunks.append(audio)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
        else:
            full_audio = np.zeros(0, dtype=np.float32)

        pcm = np.clip(full_audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        import wave

        wav_io = BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            wav_file.writeframes(pcm_bytes)
        return wav_io.getvalue()

    def synthesize_pcm(self, model_name: str, text: str, speed: float) -> bytes:
        if not text.strip():
            raise ValueError("input text is required")
        if speed < 0.25 or speed > 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")

        voice = self.load_voice(model_name)
        audio_chunks: list[np.ndarray] = []
        for chunk in voice.synthesize(
            text,
            SynthesisConfig(length_scale=1.0 / speed),
        ):
            audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
            audio_chunks.append(audio)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
        else:
            full_audio = np.zeros(0, dtype=np.float32)

        pcm = np.clip(full_audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        return pcm.tobytes()
