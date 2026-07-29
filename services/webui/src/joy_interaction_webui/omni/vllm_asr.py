"""Real sliding-window transcription through vLLM's OpenAI audio API."""

import asyncio
import io
import re
import wave
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx

from .events import AudioWindow
from .streaming_asr import TranscriptionResult


@dataclass(frozen=True)
class VllmASRConfig:
    url: str = "http://127.0.0.1:8993/v1/audio/transcriptions"
    model: str = "Qwen/Qwen3-ASR-1.7B"
    timeout_seconds: float = 30.0
    retry_attempts: int = 2
    retry_delay_seconds: float = 0.2
    max_completion_tokens: int = 64
    repetition_penalty: float = 1.1


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def extract_text(payload: dict[str, Any]) -> str:
    text = payload.get("text")
    if isinstance(text, str):
        return normalize_qwen_asr_text(text)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        content = (choices[0] or {}).get("message", {}).get("content")
        if isinstance(content, str):
            return normalize_qwen_asr_text(content)
    return ""


def normalize_qwen_asr_text(text: str) -> str:
    """Remove the language metadata emitted by Qwen3-ASR's vLLM wrapper."""
    _, marker, transcript = text.partition("<asr_text>")
    return (transcript if marker else text).strip()


def is_pathological_repetition(text: str) -> bool:
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if len(tokens) >= 8 and Counter(tokens).most_common(1)[0][1] / len(tokens) >= 0.75:
        return True
    compact = re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)
    for unit_length in range(1, min(8, len(compact) // 6) + 1):
        unit = compact[:unit_length]
        repeats, remainder = divmod(len(compact), unit_length)
        if remainder == 0 and repeats >= 6 and unit * repeats == compact:
            return True
    return False


class VllmWindowTranscriber:
    """Reuse one HTTP client per audio session and retry transient failures."""

    def __init__(
        self,
        config: VllmASRConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config or VllmASRConfig()
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self._client

    async def transcribe(self, window: AudioWindow) -> TranscriptionResult:
        wav_bytes = pcm16_to_wav(window.pcm, window.sample_rate)
        attempts = max(1, self.config.retry_attempts)
        for attempt in range(attempts):
            try:
                response = await self._get_client().post(
                    self.config.url,
                    data={
                        "model": self.config.model,
                        "max_completion_tokens": str(self.config.max_completion_tokens),
                        "temperature": "0",
                        "repetition_penalty": str(self.config.repetition_penalty),
                    },
                    files={"file": ("window.wav", wav_bytes, "audio/wav")},
                )
                response.raise_for_status()
                text = extract_text(response.json())
                return TranscriptionResult("" if is_pathological_repetition(text) else text)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(self.config.retry_delay_seconds * (2**attempt))
        raise RuntimeError("ASR retry loop exited unexpectedly")

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None
