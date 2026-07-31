from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient
from native_server import Settings, create_app


class FakeEngine:
    loaded = True

    def transcribe(self, wav_bytes: bytes) -> str:
        assert wav_bytes.startswith(b"RIFF")
        return "测试语音"


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


def test_native_transcription_is_openai_compatible() -> None:
    settings = Settings(model_path="fake", served_model_name="fake-asr")
    with TestClient(create_app(settings, FakeEngine())) as client:
        health = client.get("/health").json()
        models = client.get("/v1/models").json()
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "fake-asr"},
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
        )
    assert health["backend"] == "transformers"
    assert health["native_streaming"] is False
    assert models["data"][0]["id"] == "fake-asr"
    assert response.status_code == 200
    assert response.json() == {"text": "测试语音"}
