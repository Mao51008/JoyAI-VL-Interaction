# SPDX-License-Identifier: Apache-2.0

"""Small OpenAI-compatible JoyAI VLM server using native Transformers."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import urlopen

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger("joyvl_transformers_server")


@dataclass(frozen=True)
class Settings:
    model_path: str = "/tmp/models/jdopensource/JoyAI-VL-Interaction"
    served_model_name: str = "jdopensource/JoyAI-VL-Interaction"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    image_timeout: float = 15.0


class VLMEngine(Protocol):
    @property
    def loaded(self) -> bool: ...

    def generate(self, payload: dict[str, Any]) -> tuple[str, int, int]: ...


def settings_from_env() -> Settings:
    return Settings(
        model_path=os.getenv("MODEL_PATH", "/tmp/models/jdopensource/JoyAI-VL-Interaction"),
        served_model_name=os.getenv("SERVED_MODEL_NAME", "jdopensource/JoyAI-VL-Interaction"),
        dtype=os.getenv("MAIN_NATIVE_DTYPE", "bfloat16"),
        device=os.getenv("MAIN_NATIVE_DEVICE", "cuda:0"),
        image_timeout=float(os.getenv("MAIN_NATIVE_IMAGE_TIMEOUT", "15")),
    )


def load_image(url: str, *, timeout: float) -> Image.Image:
    if url.startswith("data:"):
        try:
            _, encoded = url.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as err:
            raise ValueError("invalid image data URL") from err
    elif url.startswith(("http://", "https://")):
        with urlopen(url, timeout=timeout) as response:
            raw = response.read()
    else:
        raise ValueError("only data:, http:// and https:// image URLs are supported")
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def normalize_messages(
    messages: list[dict[str, Any]], *, image_timeout: float
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise TypeError("message content must be a string or list")
        items: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                items.append({"type": "text", "text": str(item.get("text") or "")})
            elif item_type == "image_url":
                image_url = item.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not isinstance(url, str) or not url:
                    raise ValueError("image_url item is missing a URL")
                items.append({"type": "image", "image": load_image(url, timeout=image_timeout)})
        normalized.append({"role": role, "content": items})
    return normalized


class TransformersVLMEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._processor: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            dtype = getattr(torch, self.settings.dtype)
            logger.info("Loading native JoyAI model from %s", self.settings.model_path)
            self._processor = AutoProcessor.from_pretrained(self.settings.model_path)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.settings.model_path,
                torch_dtype=dtype,
                device_map={"": self.settings.device},
                attn_implementation="sdpa",
            )
        return self._model, self._processor

    def generate(self, payload: dict[str, Any]) -> tuple[str, int, int]:
        import torch

        model, processor = self._load()
        messages = normalize_messages(
            list(payload.get("messages") or []), image_timeout=self.settings.image_timeout
        )
        if not messages:
            raise ValueError("messages must not be empty")
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [
            item["image"]
            for message in messages
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if item.get("type") == "image"
        ]
        processor_args: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            processor_args["images"] = images
        inputs = processor(**processor_args).to(self.settings.device)
        max_tokens = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 0.0))
        generation_args: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
            generation_args["top_p"] = float(payload.get("top_p", 1.0))
        with self._inference_lock, torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_args)
        prompt_tokens = int(inputs.input_ids.shape[1])
        generated = output_ids[:, prompt_tokens:]
        completion_tokens = int(generated.shape[1])
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip(), prompt_tokens, completion_tokens


def create_app(
    settings: Settings | None = None,
    engine: VLMEngine | None = None,
) -> FastAPI:
    settings = settings or settings_from_env()
    engine = engine or TransformersVLMEngine(settings)
    app = FastAPI(title="JoyVL Native Transformers Server", version="0.1.0")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "backend": "transformers",
                "model": settings.served_model_name,
                "loaded": engine.loaded,
            }
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": settings.served_model_name,
                        "object": "model",
                        "owned_by": "local",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        requested_model = str(payload.get("model") or settings.served_model_name)
        if requested_model != settings.served_model_name:
            raise HTTPException(status_code=404, detail=f"unknown model: {requested_model}")
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="SSE streaming is not supported")
        try:
            text, prompt_tokens, completion_tokens = await asyncio.to_thread(
                engine.generate, payload
            )
        except (TypeError, ValueError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:
            logger.exception("Native VLM inference failed")
            raise HTTPException(status_code=500, detail=str(err)) from err
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": settings.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Native Transformers JoyAI VLM server")
    parser.add_argument("--host", default=os.getenv("MAIN_MODEL_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MAIN_MODEL_PORT", "7060")))
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
