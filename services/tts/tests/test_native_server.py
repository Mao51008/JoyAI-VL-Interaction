from __future__ import annotations

import json

from fastapi.testclient import TestClient
from native_server import Settings, create_app


class FakeEngine:
    loaded = True

    def synthesize(self, text: str, config: dict) -> tuple[bytes, int]:
        assert text == "你好"
        assert config["voice"] == "vivian"
        return b"\x01\x00\x02\x00", 24000


def test_native_tts_uses_existing_vllm_omni_websocket_protocol() -> None:
    settings = Settings(model_path="fake", served_model_name="fake-tts", chunk_bytes=2)
    with TestClient(create_app(settings, FakeEngine())) as client:
        health = client.get("/health").json()
        models = client.get("/v1/models").json()
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_text(json.dumps({"type": "session.config", "voice": "vivian"}))
            websocket.send_text(json.dumps({"type": "input.text", "text": "你好"}))
            websocket.send_text(json.dumps({"type": "input.done"}))
            first = websocket.receive_bytes()
            second = websocket.receive_bytes()
            audio_done = json.loads(websocket.receive_text())
            session_done = json.loads(websocket.receive_text())
    assert health["native_streaming"] is False
    assert health["generation_cancellable"] is False
    assert models["data"][0]["id"] == "fake-tts"
    assert first + second == b"\x01\x00\x02\x00"
    assert audio_done["type"] == "audio.done"
    assert session_done["type"] == "session.done"
