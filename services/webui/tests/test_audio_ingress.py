import struct

import httpx
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
    set_audio_ingress_coordinator_factory,
    setup_asr_routes,
)
from joy_interaction_webui.omni.audio_events import AudioEventDetection
from joy_interaction_webui.omni.events import AudioWindow
from joy_interaction_webui.omni.stable_prefix import StablePrefixTracker
from joy_interaction_webui.omni.streaming_asr import (
    StreamingASRConfig,
    StreamingASRCoordinator,
    TranscriptionResult,
)
from joy_interaction_webui.omni.vllm_asr import (
    VllmASRConfig,
    VllmWindowTranscriber,
    is_pathological_repetition,
    pcm16_to_wav,
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
    window = state.window_snapshot(0.04)
    assert window is not None
    assert window.last_sequence == 4
    assert window.start_ms == 120.0
    assert window.end_ms == 160.0
    assert not window.latest_voice_active


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


def test_stable_prefix_grows_monotonically_for_chinese() -> None:
    tracker = StablePrefixTracker(observations=2)

    assert tracker.update("发生")[0] == ""
    assert tracker.update("发生火")[0] == "发生"
    assert tracker.update("发生火灾")[0] == "发生火"
    stable, unstable = tracker.update("发生火警")[0:2]

    assert stable == "发生火"
    assert unstable == "警"
    tracker.reset()
    assert tracker.update("新的句子")[0] == ""


class SequenceTranscriber:
    def __init__(self, texts):
        self.texts = iter(texts)

    async def transcribe(self, window):
        return TranscriptionResult(next(self.texts), confidence=0.9)


class AlarmDetector:
    async def detect(self, window):
        if window.last_sequence == 2:
            return [AudioEventDetection("fire_alarm", 0.91, 400.0, 800.0)]
        return []


class RecoveringTranscriber:
    def __init__(self):
        self.calls = 0

    async def transcribe(self, window):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("ASR is restarting")
        return TranscriptionResult("服务已经恢复")


@pytest.mark.asyncio
async def test_streaming_coordinator_emits_partial_stable_end_and_audio_event() -> None:
    windows = iter(
        [
            AudioWindow(b"\0\0", 16000, 0, 400, 1, 1.0, True),
            AudioWindow(b"\0\0", 16000, 0, 800, 2, 1.0, True),
            AudioWindow(b"\0\0", 16000, 0, 1600, 3, 0.0, False),
        ]
    )
    events = []

    async def collect(event):
        events.append(event)

    coordinator = StreamingASRCoordinator(
        "demo",
        lambda _: next(windows),
        SequenceTranscriber(["发生", "发生火灾", "发生火灾"]),
        collect,
        detector=AlarmDetector(),
        config=StreamingASRConfig(speech_end_silence_seconds=0.8),
    )

    await coordinator.process_once()
    await coordinator.process_once()
    await coordinator.process_once()

    kinds = [event.kind for event in events]
    assert kinds == [
        "speech_start",
        "speech_partial",
        "audio_event",
        "speech_partial",
        "speech_stable",
        "speech_partial",
        "speech_stable",
        "speech_final",
        "speech_end",
    ]
    assert events[3].stable_prefix == "发生"
    assert events[2].to_dict()["label"] == "fire_alarm"
    assert events[2].to_dict()["confidence"] == 0.91


@pytest.mark.asyncio
async def test_streaming_coordinator_skips_unchanged_audio_window() -> None:
    window = AudioWindow(b"\0\0", 16000, 0, 400, 1, 0.0, False)
    coordinator = StreamingASRCoordinator(
        "demo",
        lambda _: window,
        SequenceTranscriber([]),
        lambda event: None,
    )

    await coordinator.process_once()
    await coordinator.process_once()

    assert coordinator.stats["ticks"] == 2
    assert coordinator.stats["skipped"] == 1
    assert coordinator.stats["requests"] == 0


@pytest.mark.asyncio
async def test_streaming_coordinator_recovers_after_transcriber_error() -> None:
    windows = iter(
        [
            AudioWindow(b"\0\0", 16000, 0, 400, 1, 1.0, True),
            AudioWindow(b"\0\0", 16000, 0, 800, 2, 1.0, True),
        ]
    )
    events = []

    async def collect(event):
        events.append(event)

    coordinator = StreamingASRCoordinator(
        "recover",
        lambda _: next(windows),
        RecoveringTranscriber(),
        collect,
    )

    await coordinator.process_once()
    await coordinator.process_once()

    assert [event.kind for event in events] == [
        "speech_start",
        "asr_error",
        "speech_partial",
    ]
    assert coordinator.stats["errors"] == 1
    assert events[-1].text == "服务已经恢复"


@pytest.mark.asyncio
async def test_vllm_window_transcriber_posts_wav_and_reuses_client() -> None:
    requests = []

    async def handler(request):
        body = await request.aread()
        requests.append((request, body))
        return httpx.Response(200, json={"text": "检测到火警"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transcriber = VllmWindowTranscriber(
        VllmASRConfig(
            url="http://asr.test/v1/audio/transcriptions",
            model="test-asr",
        ),
        client=client,
    )
    window = AudioWindow(b"\0\0" * 640, 16000, 0, 40, 1, 1.0, True)

    first = await transcriber.transcribe(window)
    second = await transcriber.transcribe(window)

    assert first.text == second.text == "检测到火警"
    assert len(requests) == 2
    assert all(request.url.path == "/v1/audio/transcriptions" for request, _ in requests)
    assert all(b"test-asr" in body for _, body in requests)
    assert all(b"max_completion_tokens" in body and b"64" in body for _, body in requests)
    assert pcm16_to_wav(window.pcm, 16000).startswith(b"RIFF")
    await client.aclose()


def test_pathological_asr_repetition_filter() -> None:
    assert is_pathological_repetition("Yeah! " * 20)
    assert is_pathological_repetition("报警" * 10)
    assert not is_pathological_repetition("办公室里有人说发生了火灾，请立即撤离")


@pytest.mark.asyncio
async def test_audio_ingress_can_stream_coordinator_events_to_browser() -> None:
    def factory(*, session_id, snapshot, emit):
        return StreamingASRCoordinator(
            session_id,
            snapshot,
            SequenceTranscriber(["检测到说话"]),
            emit,
            config=StreamingASRConfig(interval_seconds=0.01),
        )

    audio_ingress_sessions.clear()
    set_audio_ingress_coordinator_factory(factory)
    app = web.Application()
    setup_asr_routes(app)
    try:
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/ws/audio-ingress?session_id=test-session")
            await ws.receive_json()
            await ws.send_bytes(make_packet(sequence=1, voice_active=True))

            start = await ws.receive_json(timeout=1)
            partial = await ws.receive_json(timeout=1)

            assert start["kind"] == "speech_start"
            assert partial["kind"] == "speech_partial"
            assert partial["text"] == "检测到说话"
            await ws.close()
    finally:
        set_audio_ingress_coordinator_factory(None)
        audio_ingress_sessions.clear()
