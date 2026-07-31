from __future__ import annotations

from fastapi.testclient import TestClient
from transformers_server import Settings, create_app


class FakeEngine:
    loaded = True

    def generate(self, payload: dict) -> tuple[str, int, int]:
        assert payload["messages"][0]["content"] == "hello"
        return "world", 3, 1


def test_native_vlm_is_openai_compatible() -> None:
    settings = Settings(model_path="fake", served_model_name="fake-vlm")
    with TestClient(create_app(settings, FakeEngine())) as client:
        models = client.get("/v1/models").json()
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-vlm",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
            },
        )
    assert models["data"][0]["id"] == "fake-vlm"
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "world"
    assert payload["usage"]["total_tokens"] == 4
