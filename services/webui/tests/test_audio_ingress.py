import struct

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from joy_interaction_webui.asr import (
    AUDIO_PACKET_HEADER,
    AUDIO_PACKET_MAGIC,
    AUDIO_PACKET_VERSION,
    AudioIngressPacket,
    AudioIngressSession,
    audio_ingress_sessions,
    parse_audio_ingress_packet,
    setup_asr_routes,
)


def make_packet(
    *,
    session_id: str = "test-session",
    sequence: int = 1,
    client_monotonic_ms: float = 1234.5,
    sample_rate: int = 16000,
    samples: int = 640,
    voice_active: bool = False,
) -> bytes:
    pcm = struct.pack("<" + "h" * samples, *([123] * samples))
    session_bytes = session_id.encode("utf-8")
    header = AUDIO_PACKET_HEADER.pack(
        AUDIO_PACKET_MAGIC,
        AUDIO_PACKET_VERSION,
        1 if voice_active else 0,
        len(session_bytes),
        sequence,
        client_monotonic_ms,
        sample_rate,
        samples,
    )
    return header + session_bytes + pcm


def test_parse_audio_ingress_packet() -> None:
    packet = parse_audio_ingress_packet(make_packet())

    assert packet.session_id == "test-session"
    assert packet.sequence == 1
    assert packet.client_monotonic_ms == 1234.5
    assert packet.sample_rate == 16000
    assert len(packet.pcm) == 1280
    assert not packet.voice_active


def test_parse_audio_ingress_packet_rejects_bad_sample_count() -> None:
    data = bytearray(make_packet())
    data[24:28] = (999).to_bytes(4, "big")

    with pytest.raises(ValueError, match="sample count mismatch"):
        parse_audio_ingress_packet(bytes(data))


def test_ring_buffer_is_bounded_and_tracks_sequence_gaps() -> None:
    state = AudioIngressSession(
        session_id="test",
        sample_rate=16000,
        buffer_seconds=0.08,
    )
    pcm = b"\x00\x00" * 640

    assert state.append(AudioIngressPacket("test", 1, 0.0, 16000, pcm))
    assert state.append(AudioIngressPacket("test", 3, 80.0, 16000, pcm, True))
    assert not state.append(AudioIngressPacket("test", 2, 40.0, 16000, pcm))
    assert state.append(AudioIngressPacket("test", 4, 120.0, 16000, pcm))

    status = state.status()
    assert status["packets_received"] == 4
    assert status["packets_accepted"] == 3
    assert status["packets_dropped"] == 1
    assert status["packets_out_of_order"] == 1
    assert status["voice_active_packets"] == 1
    assert status["buffered_seconds"] == 0.08
    assert len(state.pcm_snapshot()) == 2560


def test_ring_buffer_rejects_unexpected_sample_rate() -> None:
    state = AudioIngressSession(session_id="test", sample_rate=16000)

    with pytest.raises(ValueError, match="unexpected audio sample rate"):
        state.append(AudioIngressPacket("test", 1, 0.0, 8000, b"\x00\x00"))


def test_thirty_minute_stream_keeps_only_last_thirty_seconds() -> None:
    state = AudioIngressSession(
        session_id="long-run",
        sample_rate=16000,
        buffer_seconds=30,
    )
    pcm = b"\x00\x00" * 640
    packet_count = int(30 * 60 / 0.04)

    for sequence in range(1, packet_count + 1):
        state.append(
            AudioIngressPacket(
                "long-run",
                sequence,
                (sequence - 1) * 40.0,
                16000,
                pcm,
            )
        )

    assert state.status()["buffered_seconds"] == 30.0
    assert state.buffered_bytes == 16000 * 2 * 30
    assert len(state.chunks) == 750


@pytest.mark.asyncio
async def test_audio_ingress_websocket_reports_stats_and_reconnects() -> None:
    audio_ingress_sessions.clear()
    app = web.Application()
    setup_asr_routes(app)

    async with TestClient(TestServer(app)) as client:
        ws = await client.ws_connect("/ws/audio-ingress?session_id=test-session")
        connected = await ws.receive_json()
        assert connected["message"] == "connected"
        assert connected["packet_ms"] == 40

        await ws.send_bytes(make_packet(sequence=1, voice_active=True))
        await ws.send_json({"type": "ping", "id": "stats"})
        pong = await ws.receive_json()
        assert pong["type"] == "pong"
        assert pong["stats"]["packets_accepted"] == 1
        assert pong["stats"]["voice_active_packets"] == 1
        await ws.send_json({"type": "end"})
        await ws.close()

        ws = await client.ws_connect("/ws/audio-ingress?session_id=test-session")
        await ws.receive_json()
        await ws.send_bytes(make_packet(sequence=3))
        await ws.send_json({"type": "ping", "id": "stats-2"})
        pong = await ws.receive_json()
        assert pong["stats"]["reconnects"] == 1
        assert pong["stats"]["packets_dropped"] == 1
        await ws.close()

    audio_ingress_sessions.clear()
