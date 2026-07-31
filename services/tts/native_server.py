# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni-compatible websocket shim backed by native Qwen3-TTS."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger("joyvl_tts_native")


@dataclass(frozen=True)
class Settings:
    model_path: str = "/tmp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    served_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    chunk_bytes: int = 8192


class TTSEngine(Protocol):
    @property
    def loaded(self) -> bool: ...

    def synthesize(self, text: str, config: dict[str, Any]) -> tuple[bytes, int]: ...


def settings_from_env() -> Settings:
    return Settings(
        model_path=os.getenv("TTS_MODEL_DIR", "/tmp/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        served_model_name=os.getenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        dtype=os.getenv("TTS_NATIVE_DTYPE", "bfloat16"),
        device=os.getenv("TTS_NATIVE_DEVICE", "cuda:0"),
        chunk_bytes=int(os.getenv("TTS_NATIVE_CHUNK_BYTES", "8192")),
    )


def float_audio_to_pcm16(audio: np.ndarray) -> bytes:
    samples = np.asarray(audio, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


class TransformersTTSEngine:
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
            from qwen_tts import Qwen3TTSModel

            dtype = getattr(torch, self.settings.dtype)
            logger.info("Loading native TTS model from %s", self.settings.model_path)
            self._model = Qwen3TTSModel.from_pretrained(
                self.settings.model_path,
                dtype=dtype,
                device_map=self.settings.device,
                attn_implementation="sdpa",
            )
        return self._model

    def synthesize(self, text: str, config: dict[str, Any]) -> tuple[bytes, int]:
        model = self._load()
        voice = str(config.get("voice") or "Vivian")
        instructions = str(config.get("instructions") or "")
        temperature = float(config.get("temperature", 0.7))
        max_new_tokens = int(config.get("max_new_tokens", 1024))
        with self._inference_lock:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language="Auto",
                speaker=voice,
                instruct=instructions,
                non_streaming_mode=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        if not wavs:
            raise RuntimeError("TTS model returned no waveform")
        return float_audio_to_pcm16(wavs[0]), int(sample_rate)


async def receive_session(websocket: WebSocket) -> tuple[dict[str, Any], str]:
    config: dict[str, Any] | None = None
    text = ""
    while True:
        event = json.loads(await websocket.receive_text())
        event_type = event.get("type")
        if event_type == "session.config":
            config = event
        elif event_type == "input.text":
            text += str(event.get("text") or "")
        elif event_type == "input.done":
            break
        else:
            raise ValueError(f"unsupported event: {event_type}")
    if config is None:
        raise ValueError("missing session.config")
    if not text.strip():
        raise ValueError("empty input text")
    return config, text.strip()


def create_app(
    settings: Settings | None = None,
    engine: TTSEngine | None = None,
) -> FastAPI:
    settings = settings or settings_from_env()
    engine = engine or TransformersTTSEngine(settings)
    app = FastAPI(title="JoyVL Native TTS", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "backend": "transformers",
                "model": settings.served_model_name,
                "loaded": engine.loaded,
                "native_streaming": False,
                "generation_cancellable": False,
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

    @app.websocket("/v1/audio/speech/stream")
    async def speech_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            config, text = await receive_session(websocket)
            pcm, sample_rate = await asyncio.to_thread(engine.synthesize, text, config)
            if sample_rate != 24000:
                raise RuntimeError(f"expected 24000 Hz output, got {sample_rate}")
            for offset in range(0, len(pcm), settings.chunk_bytes):
                await websocket.send_bytes(pcm[offset : offset + settings.chunk_bytes])
            await websocket.send_text(json.dumps({"type": "audio.done"}))
            await websocket.send_text(json.dumps({"type": "session.done"}))
        except WebSocketDisconnect:
            logger.info("Native TTS client disconnected")
        except Exception as err:
            logger.exception("Native TTS request failed")
            try:
                await websocket.send_text(
                    json.dumps({"type": "error", "error": str(err)}, ensure_ascii=False)
                )
            except RuntimeError:
                pass
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Native Transformers Qwen3-TTS server")
    parser.add_argument("--host", default=os.getenv("TTS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TTS_PORT", "8991")))
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
