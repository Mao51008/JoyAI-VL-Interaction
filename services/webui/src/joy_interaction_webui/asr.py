"""ASR websocket bridge for browser microphone audio."""

import asyncio
import json
import logging
import os
import struct
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

from joy_interaction_webui.omni.events import AudioWindow
from joy_interaction_webui.omni.streaming_asr import (
    StreamingASRConfig,
    StreamingASRCoordinator,
)
from joy_interaction_webui.omni.vllm_asr import VllmASRConfig, VllmWindowTranscriber

# ASR parameters
ASR_URL = os.getenv("ASR_URL", "ws://127.0.0.1:8994/ws/asr")
ASR_AUTHORIZATION = os.getenv(
    "ASR_AUTHORIZATION",
    "",
)
ASR_REQUEST_SID = os.getenv("ASR_REQUEST_SID", "browser-room")
ASR_SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
ASR_CHUNK_SECONDS = float(os.getenv("ASR_CHUNK_SECONDS", "0.04"))
ASR_CONNECT_RETRIES = int(os.getenv("ASR_CONNECT_RETRIES", "3"))
ASR_OPEN_TIMEOUT = float(os.getenv("ASR_OPEN_TIMEOUT", "10"))
ASR_RETRY_INITIAL_DELAY = float(os.getenv("ASR_RETRY_INITIAL_DELAY", "0.5"))
ASR_RETRY_MAX_DELAY = float(os.getenv("ASR_RETRY_MAX_DELAY", "5"))
ASR_FINAL_TIMEOUT = float(os.getenv("ASR_FINAL_TIMEOUT", "8.0"))
ASR_FINAL_GRACE_SECONDS = float(os.getenv("ASR_FINAL_GRACE_SECONDS", "1.2"))
ASR_RECOGNIZE_PARAMS = {
    "do_post_process": True,
    "do_partial_result": True,
    "do_punc_end_process": True,
    "do_punc_partial_process": True,
    "do_show_nbest": False,
    "do_filter_modal_part": False,
    "do_dynamic_lm": False,
    "do_server_vad": True,
    "do_semantic_vad": False,
    "continuous_decoding": True,
    "llm_reply": "",
    "agent_id": "",
    "forceend_lowerlimit": 6000,
    "forceend_upperlimit": 8000,
}
ASR_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

# Continuous microphone ingress. This path deliberately does not invoke ASR yet:
# A2 establishes a bounded, timestamped audio source; A3 consumes it with
# sliding-window/streaming recognition.
AUDIO_INGRESS_SAMPLE_RATE = int(os.getenv("AUDIO_INGRESS_SAMPLE_RATE", "16000"))
AUDIO_INGRESS_BUFFER_SECONDS = float(os.getenv("AUDIO_INGRESS_BUFFER_SECONDS", "30"))
AUDIO_INGRESS_SESSION_TTL_SECONDS = float(os.getenv("AUDIO_INGRESS_SESSION_TTL_SECONDS", "3600"))
AUDIO_PACKET_MAGIC = b"JAI1"
AUDIO_PACKET_VERSION = 1
AUDIO_PACKET_HEADER = struct.Struct(">4sBBHIdII")
CONTINUOUS_ASR_ENABLED = os.getenv("CONTINUOUS_ASR_ENABLED", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioIngressPacket:
    session_id: str
    sequence: int
    client_monotonic_ms: float
    sample_rate: int
    pcm: bytes
    voice_active: bool = False


@dataclass(frozen=True)
class BufferedAudioChunk:
    pcm: bytes
    sequence: int
    start_ms: float
    end_ms: float
    voice_active: bool


@dataclass
class AudioIngressSession:
    session_id: str
    sample_rate: int = AUDIO_INGRESS_SAMPLE_RATE
    buffer_seconds: float = AUDIO_INGRESS_BUFFER_SECONDS
    chunks: deque[BufferedAudioChunk] = field(default_factory=deque)
    buffered_bytes: int = 0
    packets_received: int = 0
    packets_accepted: int = 0
    packets_dropped: int = 0
    packets_out_of_order: int = 0
    voice_active_packets: int = 0
    reconnects: int = 0
    active_connections: int = 0
    last_sequence: int | None = None
    first_client_monotonic_ms: float | None = None
    first_server_monotonic_ms: float | None = None
    last_client_monotonic_ms: float | None = None
    clock_drift_ms: float | None = None
    last_seen_server_monotonic: float = field(default_factory=time.monotonic)

    @property
    def max_buffer_bytes(self) -> int:
        return max(2, int(self.sample_rate * self.buffer_seconds) * 2)

    def connection_opened(self) -> None:
        if self.packets_received or self.active_connections:
            self.reconnects += 1
        self.active_connections += 1
        self.last_seen_server_monotonic = time.monotonic()

    def connection_closed(self) -> None:
        self.active_connections = max(0, self.active_connections - 1)
        self.last_seen_server_monotonic = time.monotonic()

    def append(self, packet: AudioIngressPacket) -> bool:
        self.packets_received += 1
        now = time.monotonic()
        self.last_seen_server_monotonic = now

        if packet.sample_rate != self.sample_rate:
            raise ValueError(
                f"unexpected audio sample rate {packet.sample_rate}; expected {self.sample_rate}"
            )
        if len(packet.pcm) % 2:
            raise ValueError("PCM16 payload byte length must be even")

        if self.last_sequence is not None:
            if packet.sequence <= self.last_sequence:
                self.packets_out_of_order += 1
                return False
            self.packets_dropped += max(0, packet.sequence - self.last_sequence - 1)
        self.last_sequence = packet.sequence

        if self.first_client_monotonic_ms is None:
            self.first_client_monotonic_ms = packet.client_monotonic_ms
            self.first_server_monotonic_ms = now * 1000
        self.last_client_monotonic_ms = packet.client_monotonic_ms
        if self.first_server_monotonic_ms is not None:
            client_elapsed = packet.client_monotonic_ms - self.first_client_monotonic_ms
            server_elapsed = now * 1000 - self.first_server_monotonic_ms
            self.clock_drift_ms = client_elapsed - server_elapsed

        duration_ms = len(packet.pcm) / (self.sample_rate * 2) * 1000
        self.chunks.append(
            BufferedAudioChunk(
                pcm=packet.pcm,
                sequence=packet.sequence,
                start_ms=packet.client_monotonic_ms,
                end_ms=packet.client_monotonic_ms + duration_ms,
                voice_active=packet.voice_active,
            )
        )
        self.buffered_bytes += len(packet.pcm)
        if packet.voice_active:
            self.voice_active_packets += 1
        while self.buffered_bytes > self.max_buffer_bytes and self.chunks:
            self.buffered_bytes -= len(self.chunks.popleft().pcm)
        self.packets_accepted += 1
        return True

    def pcm_snapshot(self) -> bytes:
        return b"".join(chunk.pcm for chunk in self.chunks)

    def window_snapshot(self, seconds: float) -> AudioWindow | None:
        if not self.chunks:
            return None
        byte_limit = max(2, int(seconds * self.sample_rate) * 2)
        selected: list[BufferedAudioChunk] = []
        selected_bytes = 0
        for chunk in reversed(self.chunks):
            selected.append(chunk)
            selected_bytes += len(chunk.pcm)
            if selected_bytes >= byte_limit:
                break
        selected.reverse()
        pcm = b"".join(chunk.pcm for chunk in selected)
        if len(pcm) > byte_limit:
            pcm = pcm[-byte_limit:]
        voice_count = sum(chunk.voice_active for chunk in selected)
        return AudioWindow(
            pcm=pcm,
            sample_rate=self.sample_rate,
            start_ms=selected[0].start_ms,
            end_ms=selected[-1].end_ms,
            last_sequence=selected[-1].sequence,
            voice_ratio=voice_count / len(selected),
            latest_voice_active=selected[-1].voice_active,
        )

    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "buffer_limit_seconds": self.buffer_seconds,
            "buffered_seconds": round(self.buffered_bytes / (self.sample_rate * 2), 3),
            "packets_received": self.packets_received,
            "packets_accepted": self.packets_accepted,
            "packets_dropped": self.packets_dropped,
            "packets_out_of_order": self.packets_out_of_order,
            "voice_active_packets": self.voice_active_packets,
            "reconnects": self.reconnects,
            "active_connections": self.active_connections,
            "last_sequence": self.last_sequence,
            "last_client_monotonic_ms": self.last_client_monotonic_ms,
            "clock_drift_ms": (
                round(self.clock_drift_ms, 3) if self.clock_drift_ms is not None else None
            ),
        }


audio_ingress_sessions: dict[str, AudioIngressSession] = {}
audio_ingress_coordinator_factory = None


def set_audio_ingress_coordinator_factory(factory) -> None:
    """Inject the streaming recognizer backend; A3.2 supplies the real ASR client."""
    global audio_ingress_coordinator_factory
    audio_ingress_coordinator_factory = factory


def configure_continuous_asr_from_env() -> None:
    if not CONTINUOUS_ASR_ENABLED:
        return
    transcriber_config = VllmASRConfig(
        url=os.getenv(
            "CONTINUOUS_ASR_URL",
            "http://127.0.0.1:8993/v1/audio/transcriptions",
        ),
        model=os.getenv("CONTINUOUS_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
        timeout_seconds=float(os.getenv("CONTINUOUS_ASR_TIMEOUT_SECONDS", "30")),
        retry_attempts=int(os.getenv("CONTINUOUS_ASR_RETRY_ATTEMPTS", "2")),
        retry_delay_seconds=float(os.getenv("CONTINUOUS_ASR_RETRY_DELAY_SECONDS", "0.2")),
    )
    coordinator_config = StreamingASRConfig(
        interval_seconds=float(os.getenv("CONTINUOUS_ASR_INTERVAL_SECONDS", "0.4")),
        window_seconds=float(os.getenv("CONTINUOUS_ASR_WINDOW_SECONDS", "6")),
        speech_end_silence_seconds=float(
            os.getenv("CONTINUOUS_ASR_SPEECH_END_SILENCE_SECONDS", "0.8")
        ),
        stable_observations=int(os.getenv("CONTINUOUS_ASR_STABLE_OBSERVATIONS", "2")),
    )

    def factory(*, session_id, snapshot, emit):
        return StreamingASRCoordinator(
            session_id,
            snapshot,
            VllmWindowTranscriber(transcriber_config),
            emit,
            config=coordinator_config,
        )

    set_audio_ingress_coordinator_factory(factory)
    logger.info(
        "Continuous ASR enabled: model=%s url=%s interval=%.3fs window=%.1fs",
        transcriber_config.model,
        transcriber_config.url,
        coordinator_config.interval_seconds,
        coordinator_config.window_seconds,
    )


configure_continuous_asr_from_env()


def parse_audio_ingress_packet(data: bytes) -> AudioIngressPacket:
    if len(data) < AUDIO_PACKET_HEADER.size:
        raise ValueError(
            f"audio ingress packet shorter than {AUDIO_PACKET_HEADER.size}-byte header"
        )
    magic, version, flags, session_id_bytes, sequence, client_ms, sample_rate, sample_count = (
        AUDIO_PACKET_HEADER.unpack(data[: AUDIO_PACKET_HEADER.size])
    )
    if magic != AUDIO_PACKET_MAGIC:
        raise ValueError("invalid audio ingress packet magic")
    if version != AUDIO_PACKET_VERSION:
        raise ValueError(f"unsupported audio ingress packet version {version}")
    session_start = AUDIO_PACKET_HEADER.size
    session_end = session_start + session_id_bytes
    if len(data) < session_end:
        raise ValueError("audio ingress packet has truncated session_id")
    try:
        session_id = data[session_start:session_end].decode("utf-8")
    except UnicodeDecodeError as err:
        raise ValueError("audio ingress packet session_id is not UTF-8") from err
    pcm = data[session_end:]
    if len(pcm) != sample_count * 2:
        raise ValueError(
            f"audio ingress sample count mismatch: header={sample_count}, bytes={len(pcm)}"
        )
    return AudioIngressPacket(
        session_id=session_id,
        sequence=sequence,
        client_monotonic_ms=client_ms,
        sample_rate=sample_rate,
        pcm=pcm,
        voice_active=bool(flags & 0x01),
    )


def prune_audio_ingress_sessions() -> None:
    cutoff = time.monotonic() - AUDIO_INGRESS_SESSION_TTL_SECONDS
    expired = [
        session_id
        for session_id, state in audio_ingress_sessions.items()
        if not state.active_connections and state.last_seen_server_monotonic < cutoff
    ]
    for session_id in expired:
        audio_ingress_sessions.pop(session_id, None)


def get_audio_ingress_session(session_id: str) -> AudioIngressSession:
    prune_audio_ingress_sessions()
    state = audio_ingress_sessions.get(session_id)
    if state is None:
        state = AudioIngressSession(session_id=session_id)
        audio_ingress_sessions[session_id] = state
    return state


def get_audio_ingress_pcm(session_id: str) -> bytes:
    state = audio_ingress_sessions.get(session_id)
    return state.pcm_snapshot() if state is not None else b""


def cleanup_audio_ingress_session(session_id: str) -> bool:
    return audio_ingress_sessions.pop(session_id, None) is not None


def mask_secret(value):
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def build_asr_headers():
    request_params = {
        "sid": ASR_REQUEST_SID,
        "reqid": str(uuid.uuid1()),
        "sample_rate": ASR_SAMPLE_RATE,
    }
    return {
        "authorization": ASR_AUTHORIZATION,
        "request": json.dumps(request_params),
        "recognize": json.dumps(ASR_RECOGNIZE_PARAMS),
    }


def retry_asr_delay(attempt):
    return min(ASR_RETRY_INITIAL_DELAY * (2**attempt), ASR_RETRY_MAX_DELAY)


def is_retryable_asr_connect_error(err):
    if isinstance(err, (asyncio.TimeoutError, OSError, aiohttp.ClientConnectionError)):
        return True
    status = getattr(err, "status", None)
    return status in ASR_RETRYABLE_STATUS_CODES


async def connect_asr(session_id):
    if not ASR_URL:
        raise RuntimeError("ASR_URL is not configured")

    attempt = 0
    while True:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        try:
            logger.info(
                "[%s] ASR connect attempt %s url=%s authorization=%s",
                session_id,
                attempt + 1,
                ASR_URL,
                mask_secret(ASR_AUTHORIZATION),
            )
            asr_ws = await session.ws_connect(
                ASR_URL,
                headers=build_asr_headers(),
                timeout=ASR_OPEN_TIMEOUT,
                heartbeat=20,
                max_msg_size=0,
            )
            return session, asr_ws
        except Exception as err:
            await session.close()
            retryable = is_retryable_asr_connect_error(err)
            retries_left = ASR_CONNECT_RETRIES < 0 or attempt < ASR_CONNECT_RETRIES
            logger.warning(
                "[%s] ASR connect attempt %s failed retryable=%s retries_left=%s: %s",
                session_id,
                attempt + 1,
                retryable,
                retries_left,
                err,
            )
            if not retryable or not retries_left:
                raise
            delay = retry_asr_delay(attempt)
            attempt += 1
            await asyncio.sleep(delay)


def pack_asr_audio(seqid, audio, is_final=False):
    packet_seqid = -abs(seqid) if is_final else seqid
    return struct.pack(">iii", packet_seqid, 0, 0) + audio


def extract_asr_result(payload):
    asr_response = payload.get("asr_response") or {}
    event_type = asr_response.get("event_type", "")
    recognition = asr_response.get("recognition_result") or {}
    hypotheses = recognition.get("hypothesis") or []
    first = hypotheses[0] if hypotheses else {}
    text = first.get("text", "")
    if event_type not in {"IS_PARTIAL", "IS_FINAL", "IS_END"}:
        text = ""
    return {
        "type": "result",
        "event": event_type,
        "mid": payload.get("mid", ""),
        "text": text,
        "confidence": first.get("confidence"),
        "final": event_type in {"IS_FINAL", "IS_END"},
        "code": payload.get("code"),
        "msg": payload.get("msg", ""),
    }


def make_asr_synthetic_final(mid, text, message):
    return {
        "type": "result",
        "event": "IS_FINAL",
        "mid": mid or "",
        "text": text,
        "confidence": None,
        "final": True,
        "code": 0,
        "msg": message,
        "synthetic": True,
    }


async def send_asr_client_json(client_ws, payload):
    if not client_ws.closed:
        await client_ws.send_str(json.dumps(payload, ensure_ascii=False))


async def forward_asr_audio(session_id, client_ws, asr_ws, client_end_event):
    seqid = 1
    pending = bytearray()
    chunk_bytes = max(2, int(ASR_SAMPLE_RATE * ASR_CHUNK_SECONDS) * 2)
    final_sent = False
    sent_bytes = 0

    async def send_audio(audio, is_final=False):
        nonlocal seqid, sent_bytes
        await asr_ws.send_bytes(pack_asr_audio(seqid, audio, is_final=is_final))
        sent_bytes += len(audio)
        seqid += 1

    async def flush_final():
        nonlocal final_sent
        client_end_event.set()
        if final_sent or asr_ws.closed:
            return
        final_sent = True
        while pending:
            audio = bytes(pending[:chunk_bytes])
            del pending[:chunk_bytes]
            await send_audio(audio)
        await send_audio(b"", is_final=True)
        logger.info(
            "[%s] ASR final audio sent audio_seconds=%.3f",
            session_id,
            sent_bytes / (ASR_SAMPLE_RATE * 2),
        )

    async for msg in client_ws:
        if msg.type == web.WSMsgType.BINARY:
            pending.extend(msg.data)
            while len(pending) >= chunk_bytes:
                await send_audio(bytes(pending[:chunk_bytes]))
                del pending[:chunk_bytes]
        elif msg.type == web.WSMsgType.TEXT:
            try:
                control = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if control.get("type") == "ping":
                await send_asr_client_json(
                    client_ws,
                    {
                        "type": "pong",
                        "id": control.get("id"),
                        "client_ts": control.get("client_ts"),
                        "server_ts": time.time(),
                    },
                )
            elif control.get("type") in {"end", "segment_end"}:
                await flush_final()
                return
        elif msg.type in {web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED}:
            break
        elif msg.type == web.WSMsgType.ERROR:
            raise client_ws.exception() or RuntimeError("ASR client websocket error")

    await flush_final()


async def forward_asr_results(
    session_id,
    client_ws,
    asr_ws,
    stop_on_final=True,
    client_end_event=None,
):
    last_text = ""
    ending_mid = None
    while True:
        timeout = ASR_FINAL_GRACE_SECONDS if ending_mid else None
        try:
            msg = await asr_ws.receive(timeout=timeout)
        except asyncio.TimeoutError:
            if last_text:
                await send_asr_client_json(
                    client_ws,
                    make_asr_synthetic_final(
                        ending_mid,
                        last_text,
                        "synthetic final after ASR end timeout",
                    ),
                )
            if stop_on_final or (client_end_event and client_end_event.is_set()):
                return
            ending_mid = None
            continue

        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            result = extract_asr_result(payload)
            logger.debug("[%s] ASR result: %s", session_id, result)
            if ending_mid and result["mid"] and result["mid"] != ending_mid:
                if last_text:
                    await send_asr_client_json(
                        client_ws,
                        make_asr_synthetic_final(
                            ending_mid,
                            last_text,
                            "synthetic final before next ASR segment",
                        ),
                    )
                if stop_on_final or (client_end_event and client_end_event.is_set()):
                    return
                ending_mid = None
            await send_asr_client_json(client_ws, result)
            if result["text"]:
                last_text = result["text"]
            if result["final"] and (
                stop_on_final or (client_end_event and client_end_event.is_set())
            ):
                return
            if result["event"] == "IS_IPU_END":
                ending_mid = result["mid"] or "unknown"
        elif msg.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            return
        elif msg.type == aiohttp.WSMsgType.ERROR:
            raise asr_ws.exception() or RuntimeError("ASR upstream websocket error")


async def asr_websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    session_id = request.query.get("session_id", "").strip() or uuid.uuid4().hex[:8]
    continuous_results = request.query.get("continuous") == "1"
    client_end_event = asyncio.Event()
    asr_session = None
    asr_ws = None
    logger.info("[%s] Browser ASR websocket connected", session_id)

    try:
        asr_session, asr_ws = await connect_asr(session_id)
        await send_asr_client_json(
            ws,
            {"type": "status", "message": "connected", "sample_rate": ASR_SAMPLE_RATE},
        )

        audio_task = asyncio.create_task(
            forward_asr_audio(session_id, ws, asr_ws, client_end_event)
        )
        result_task = asyncio.create_task(
            forward_asr_results(
                session_id,
                ws,
                asr_ws,
                stop_on_final=not continuous_results,
                client_end_event=client_end_event,
            )
        )
        done, pending = await asyncio.wait(
            {audio_task, result_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if audio_task in done and not result_task.done():
            try:
                await asyncio.wait_for(result_task, timeout=ASR_FINAL_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("[%s] ASR final result timeout", session_id)
                result_task.cancel()
        else:
            for task in pending:
                task.cancel()
        for task in done:
            task.result()
    except Exception as err:
        logger.exception("[%s] ASR websocket failed", session_id)
        try:
            await send_asr_client_json(ws, {"type": "error", "message": f"ASR failed: {err}"})
        except Exception:
            pass
    finally:
        if asr_ws is not None and not asr_ws.closed:
            await asr_ws.close()
        if asr_session is not None:
            await asr_session.close()
        if not ws.closed:
            await ws.close()
        logger.info("[%s] Browser ASR websocket closed", session_id)

    return ws


async def audio_ingress_websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await ws.prepare(request)

    session_id = request.query.get("session_id", "").strip() or uuid.uuid4().hex[:8]
    state = get_audio_ingress_session(session_id)
    coordinator: StreamingASRCoordinator | None = None
    state.connection_opened()
    logger.info("[%s] Continuous audio ingress connected", session_id)
    await send_asr_client_json(
        ws,
        {
            "type": "status",
            "message": "connected",
            "sample_rate": state.sample_rate,
            "packet_ms": 40,
            "buffer_seconds": state.buffer_seconds,
        },
    )
    if audio_ingress_coordinator_factory is not None:

        async def emit_audio_event(event):
            if not ws.closed:
                await send_asr_client_json(ws, event.to_dict())

        coordinator = audio_ingress_coordinator_factory(
            session_id=session_id,
            snapshot=state.window_snapshot,
            emit=emit_audio_event,
        )
        coordinator.start()

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                try:
                    packet = parse_audio_ingress_packet(msg.data)
                    if packet.session_id != session_id:
                        raise ValueError(
                            "audio ingress packet session_id does not match websocket session"
                        )
                    state.append(packet)
                except ValueError as err:
                    await send_asr_client_json(
                        ws,
                        {"type": "error", "message": str(err)},
                    )
            elif msg.type == web.WSMsgType.TEXT:
                try:
                    control = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "ping":
                    await send_asr_client_json(
                        ws,
                        {
                            "type": "pong",
                            "id": control.get("id"),
                            "client_ts": control.get("client_ts"),
                            "server_ts": time.time(),
                            "stats": state.status(),
                        },
                    )
                elif control.get("type") == "end":
                    break
            elif msg.type in {
                web.WSMsgType.CLOSE,
                web.WSMsgType.CLOSING,
                web.WSMsgType.CLOSED,
            }:
                break
            elif msg.type == web.WSMsgType.ERROR:
                raise ws.exception() or RuntimeError("audio ingress websocket error")
    except Exception:
        logger.exception("[%s] Continuous audio ingress failed", session_id)
    finally:
        if coordinator is not None:
            await coordinator.stop()
        state.connection_closed()
        if not ws.closed:
            await ws.close()
        logger.info(
            "[%s] Continuous audio ingress closed stats=%s",
            session_id,
            state.status(),
        )

    return ws


async def audio_ingress_status_handler(request):
    session_id = request.query.get("session_id", "").strip()
    if session_id:
        state = audio_ingress_sessions.get(session_id)
        if state is None:
            return web.json_response(
                {"error": "unknown session_id", "session_id": session_id},
                status=404,
            )
        return web.json_response(state.status())
    prune_audio_ingress_sessions()
    return web.json_response(
        {"sessions": [state.status() for state in audio_ingress_sessions.values()]}
    )


def setup_asr_routes(app):
    app.router.add_get("/ws/asr", asr_websocket_handler)
    app.router.add_get("/ws/audio-ingress", audio_ingress_websocket_handler)
    app.router.add_get("/api/audio-ingress/status", audio_ingress_status_handler)
