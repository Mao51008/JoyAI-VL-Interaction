import asyncio

import pytest
from fastapi import WebSocketDisconnect

from tts_adapter import forward_upstream_audio


class DisconnectingClient:
    async def receive(self):
        return {"type": "websocket.disconnect", "code": 1000}

    async def send_bytes(self, data):
        raise AssertionError(f"unexpected audio: {data!r}")

    async def send_text(self, data):
        raise AssertionError(f"unexpected event: {data!r}")


class BlockingUpstream:
    def __init__(self):
        self.recv_cancelled = asyncio.Event()
        self.transport = FakeTransport()

    async def recv(self):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.recv_cancelled.set()
            raise


class FakeTransport:
    def __init__(self):
        self.aborted = False

    def abort(self):
        self.aborted = True


@pytest.mark.asyncio
async def test_downstream_disconnect_cancels_blocked_upstream_receive() -> None:
    upstream = BlockingUpstream()

    with pytest.raises(WebSocketDisconnect):
        await forward_upstream_audio(
            client_ws=DisconnectingClient(),
            upstream_ws=upstream,
            idle_timeout=1,
        )

    assert upstream.recv_cancelled.is_set()
    assert upstream.transport.aborted is True
