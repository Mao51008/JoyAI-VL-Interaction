# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible Qwen3-ASR server backed by native Transformers."""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import threading
import wave
from dataclasses import dataclass
from typing import Annotated, Any, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("joyvl_asr_native")


@dataclass(frozen=True)
class Settings:
    model_path: str = "/tmp/models/Qwen3-ASR-1.7B"
    served_model_name: str = "Qwen/Qwen3-ASR-1.7B"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    max_inference_batch_size: int = 8
    max_new_tokens: int = 512


class ASREngine(Protocol):
    @property
    def loaded(self) -> bool: ...

    def transcribe(self, wav_bytes: bytes) -> str: ...


def settings_from_env() -> Settings:
    return Settings(
        model_path=os.getenv("ASR_MODEL_DIR", "/tmp/models/Qwen3-ASR-1.7B"),
        served_model_name=os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B"),
        dtype=os.getenv("ASR_NATIVE_DTYPE", "bfloat16"),
        device=os.getenv("ASR_NATIVE_DEVICE", "cuda:0"),
        max_inference_batch_size=int(os.getenv("ASR_NATIVE_MAX_BATCH_SIZE", "8")),
        max_new_tokens=int(os.getenv("ASR_NATIVE_MAX_NEW_TOKENS", "512")),
    )


def decode_pcm16_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("only mono WAV input is supported")
        if wav_file.getsampwidth() != 2:
            raise ValueError("only PCM16 WAV input is supported")
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


class TransformersASREngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            import torch
            from qwen_asr import Qwen3ASRModel

            dtype = getattr(torch, self.settings.dtype)
            logger.info("Loading native ASR model from %s", self.settings.model_path)
            self._model = Qwen3ASRModel.from_pretrained(
                self.settings.model_path,
                dtype=dtype,
                device_map=self.settings.device,
                attn_implementation="sdpa",
                max_inference_batch_size=self.settings.max_inference_batch_size,
                max_new_tokens=self.settings.max_new_tokens,
            )
        return self._model

    def transcribe(self, wav_bytes: bytes) -> str:
        audio, sample_rate = decode_pcm16_wav(wav_bytes)
        model = self._load()
        with self._inference_lock:
            results = model.transcribe(audio=(audio, sample_rate), language=None)
        if not results:
            return ""
        return str(results[0].text).strip()


def create_app(
    settings: Settings | None = None,
    engine: ASREngine | None = None,
) -> FastAPI:
    settings = settings or settings_from_env()
    engine = engine or TransformersASREngine(settings)
    app = FastAPI(title="JoyVL Native ASR", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "backend": "transformers",
                "model": settings.served_model_name,
                "loaded": engine.loaded,
                "native_streaming": False,
            }
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": settings.served_model_name, "object": "model"}],
            }
        )

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "",
    ) -> JSONResponse:
        if model and model not in {settings.served_model_name, settings.model_path}:
            raise HTTPException(status_code=404, detail=f"unknown model: {model}")
        wav_bytes = await file.read()
        if not wav_bytes:
            raise HTTPException(status_code=400, detail="empty audio file")
        try:
            text = await asyncio.to_thread(engine.transcribe, wav_bytes)
        except (ValueError, wave.Error) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:
            logger.exception("Native ASR transcription failed")
            raise HTTPException(status_code=500, detail=str(err)) from err
        return JSONResponse({"text": text})

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Native Transformers Qwen3-ASR server")
    parser.add_argument("--host", default=os.getenv("ASR_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ASR_PORT", "8993")))
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
