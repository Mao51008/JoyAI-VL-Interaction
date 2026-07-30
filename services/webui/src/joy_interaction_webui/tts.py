
"""TTS bridge for streaming VLM text responses as browser-playable PCM audio."""

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
import wave

import aiohttp
from aiohttp import web

from .omni.orchestrator import get_or_create_orchestrator
from .omni.timeline import TimelineEvent

# TTS parameters
TTS_URL = os.getenv("TTS_URL", "ws://127.0.0.1:8992/ws/tts")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
TTS_VOICE = os.getenv("TTS_VOICE", "default")
TTS_EMOTION = os.getenv("TTS_EMOTION", "{{高兴}}")
TTS_CHUNK_SIZE = int(os.getenv("TTS_CHUNK_SIZE", "12"))
TTS_OPEN_TIMEOUT = float(os.getenv("TTS_OPEN_TIMEOUT", "10.0"))
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "120.0"))
TTS_STREAM_IDLE_TIMEOUT = float(os.getenv("TTS_STREAM_IDLE_TIMEOUT", "60.0"))
TTS_CANCEL_TIMEOUT = float(os.getenv("TTS_CANCEL_TIMEOUT", "1.0"))
TTS_MAX_TEXT_CHARS = int(os.getenv("TTS_MAX_TEXT_CHARS", "2000"))
TTS_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.7"))
TTS_MAX_TOKENS = int(os.getenv("TTS_MAX_TOKENS", "1024"))
TTS_INSTRUCTIONS = os.getenv(
    "TTS_INSTRUCTIONS",
    (
        "Please speak at a slightly faster pace, around 1.2x normal speed, "
        "while keeping pronunciation clear and natural."
    ),
)
TTS_PROXY = os.getenv("TTS_PROXY", "").strip() or None
TTS_TRUST_ENV = os.getenv("TTS_TRUST_ENV", "1").lower() not in {
    "0",
    "false",
    "no",
}

logger = logging.getLogger(__name__)

_active_tts_generations: dict[str, tuple[str, asyncio.Task]] = {}


def register_tts_generation(
    session_id: str,
    generation_id: str,
    task: asyncio.Task,
) -> None:
    _active_tts_generations[session_id] = (generation_id, task)


def get_active_tts_generation(session_id: str) -> str | None:
    active = _active_tts_generations.get(session_id)
    if active is None or active[1].done():
        return None
    return active[0]


def unregister_tts_generation(session_id: str, generation_id: str) -> None:
    active = _active_tts_generations.get(session_id)
    if active is not None and active[0] == generation_id:
        _active_tts_generations.pop(session_id, None)


def iter_text_chunks(text: str, chunk_size: int = TTS_CHUNK_SIZE):
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def normalize_tts_text(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    if TTS_MAX_TEXT_CHARS > 0:
        return normalized[:TTS_MAX_TEXT_CHARS]
    return normalized


def build_tts_config(sample_rate: int = TTS_SAMPLE_RATE, voice: str = TTS_VOICE) -> dict:
    return {
        "config": {
            "modalities": ["text", "audio"],
            "voice": voice,
            "instructions": TTS_INSTRUCTIONS,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "sample_rate": sample_rate,
            "temperature": TTS_TEMPERATURE,
            "max_tokens": TTS_MAX_TOKENS,
        }
    }


async def send_tts_text(websocket, text: str, chunk_size: int, emotion: str, reqid: str):
    for chunk in iter_text_chunks(text, chunk_size):
        await websocket.send_str(
            json.dumps(
                {
                    "type": "input_text.append",
                    "text": chunk,
                    "emotion": emotion,
                    "reqid": reqid,
                },
                ensure_ascii=False,
            )
        )
        await asyncio.sleep(0.05)

    await websocket.send_str(json.dumps({"type": "input_text.commit", "reqid": reqid}))


async def receive_tts_pcm(websocket) -> bytes:
    pcm = bytearray()

    while True:
        msg = await websocket.receive()

        if msg.type == aiohttp.WSMsgType.BINARY:
            pcm.extend(msg.data)
            continue

        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                message = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("[tts] skipped non-JSON text event")
                continue

            event_type = message.get("type")
            if event_type == "response.audio.delta":
                audio_b64 = message.get("delta", "")
                if audio_b64:
                    pcm.extend(base64.b64decode(audio_b64))
            elif event_type == "response.done":
                return bytes(pcm)
            elif event_type == "error":
                raise RuntimeError(f"TTS server error: {message}")
            else:
                logger.debug("[tts] event: %s", event_type)
            continue

        if msg.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            return bytes(pcm)

        if msg.type == aiohttp.WSMsgType.ERROR:
            raise websocket.exception() or RuntimeError("TTS upstream websocket error")


async def forward_tts_stream(
    websocket,
    client_ws,
    idle_timeout: float = TTS_STREAM_IDLE_TIMEOUT,
) -> int:
    total_audio_bytes = 0

    while True:
        try:
            msg = await websocket.receive(timeout=idle_timeout)
        except asyncio.TimeoutError as err:
            raise TimeoutError(f"TTS stream idle timeout after {idle_timeout}s") from err

        if msg.type == aiohttp.WSMsgType.BINARY:
            await client_ws.send_bytes(msg.data)
            total_audio_bytes += len(msg.data)
            continue

        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                message = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("[tts] skipped non-JSON text event")
                continue

            event_type = message.get("type")
            if event_type == "response.audio.delta":
                audio_b64 = message.get("delta", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    await client_ws.send_bytes(audio_bytes)
                    total_audio_bytes += len(audio_bytes)
            elif event_type == "response.done":
                return total_audio_bytes
            elif event_type == "error":
                raise RuntimeError(f"TTS server error: {message}")
            else:
                logger.debug("[tts] event: %s", event_type)
            continue

        if msg.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            return total_audio_bytes

        if msg.type == aiohttp.WSMsgType.ERROR:
            raise websocket.exception() or RuntimeError("TTS upstream websocket error")


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


async def synthesize_tts_pcm(
    text: str,
    *,
    url: str = TTS_URL,
    sample_rate: int = TTS_SAMPLE_RATE,
    voice: str = TTS_VOICE,
    open_timeout: float = TTS_OPEN_TIMEOUT,
    timeout: float = TTS_TIMEOUT,
    chunk_size: int = TTS_CHUNK_SIZE,
    emotion: str = TTS_EMOTION,
    reqid: str | None = None,
) -> bytes:
    text = normalize_tts_text(text)
    if not text:
        raise ValueError("text must not be empty")

    reqid = reqid or uuid.uuid4().hex
    logger.info("[tts] synthesize reqid=%s chars=%s url=%s", reqid, len(text), url)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None),
        trust_env=TTS_TRUST_ENV,
    ) as session:
        websocket = await session.ws_connect(
            url,
            timeout=open_timeout,
            heartbeat=20,
            max_msg_size=0,
            proxy=TTS_PROXY,
        )
        try:
            await websocket.send_str(json.dumps(build_tts_config(sample_rate, voice)))
            await send_tts_text(websocket, text, chunk_size, emotion, reqid)
            pcm = await asyncio.wait_for(receive_tts_pcm(websocket), timeout=timeout)
        finally:
            if not websocket.closed:
                await websocket.close()

    if not pcm:
        raise RuntimeError("TTS returned no audio")
    logger.info("[tts] synthesized reqid=%s audio_bytes=%s", reqid, len(pcm))
    return pcm


async def synthesize_tts_wav(text: str, **kwargs) -> tuple[bytes, int]:
    sample_rate = int(kwargs.pop("sample_rate", None) or TTS_SAMPLE_RATE)
    pcm = await synthesize_tts_pcm(text, sample_rate=sample_rate, **kwargs)
    return pcm16_to_wav_bytes(pcm, sample_rate), len(pcm)


async def run_tts_stream_request(client_ws, data):
    upstream_session = None
    upstream_ws = None
    reqid = data.get("request_id") or data.get("reqid")
    session_id = str(data.get("session_id") or "web")
    generation_id = str(data.get("generation_id") or "").strip()
    timeline_start_ms = None
    timeline_status = "failed"
    timeline_text = ""

    try:
        text = normalize_tts_text(data.get("text", ""))
        timeline_text = text
        if not text:
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": "Missing text"}
            )
            return

        try:
            sample_rate = int(data.get("sample_rate") or TTS_SAMPLE_RATE)
        except (TypeError, ValueError):
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": "Invalid sample_rate"}
            )
            return

        url = data.get("url") or TTS_URL
        voice = data.get("voice") or TTS_VOICE
        emotion = data.get("emotion") or TTS_EMOTION
        reqid = reqid or f"{data.get('session_id') or 'web'}-{uuid.uuid4().hex[:12]}"
        generation_id = generation_id or f"{session_id}-{uuid.uuid4().hex}"
        timeline_start_ms = time.monotonic() * 1000
        orchestrator = get_or_create_orchestrator(session_id)
        orchestrator.start()
        await orchestrator.record_event(
            TimelineEvent(
                session_id=session_id,
                modality="audio",
                kind="tts_playback",
                start_ms=timeline_start_ms,
                end_ms=timeline_start_ms,
                payload={
                    "source": "system_output",
                    "request_id": reqid,
                    "generation_id": generation_id,
                    "status": "playing",
                    "text": timeline_text,
                },
                priority=30,
            )
        )

        await client_ws.send_json(
            {
                "type": "start",
                "request_id": reqid,
                "generation_id": generation_id,
                "format": "pcm16",
                "sample_rate": sample_rate,
                "channels": 1,
                "reqid": reqid,
            }
        )

        logger.info("[tts] stream reqid=%s chars=%s url=%s", reqid, len(text), url)
        upstream_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None),
            trust_env=TTS_TRUST_ENV,
        )
        upstream_ws = await upstream_session.ws_connect(
            url,
            timeout=TTS_OPEN_TIMEOUT,
            heartbeat=20,
            max_msg_size=0,
            proxy=TTS_PROXY,
        )
        await upstream_ws.send_str(json.dumps(build_tts_config(sample_rate, voice)))
        await send_tts_text(upstream_ws, text, TTS_CHUNK_SIZE, emotion, reqid)
        total_audio_bytes = await forward_tts_stream(upstream_ws, client_ws)
        await client_ws.send_json(
            {
                "type": "done",
                "request_id": reqid,
                "generation_id": generation_id,
                "reqid": reqid,
                "audio_bytes": total_audio_bytes,
            }
        )
        logger.info("[tts] stream done reqid=%s audio_bytes=%s", reqid, total_audio_bytes)
        timeline_status = "done"
    except asyncio.CancelledError:
        timeline_status = "cancelled"
        logger.info("[tts] stream cancelled reqid=%s", reqid)
        if not client_ws.closed:
            try:
                await client_ws.send_json(
                    {
                        "type": "stopped",
                        "request_id": reqid,
                        "reqid": reqid,
                        "generation_id": generation_id,
                    }
                )
            except (ConnectionResetError, RuntimeError):
                pass
        raise
    except asyncio.TimeoutError:
        timeline_status = "timeout"
        logger.warning("[tts] stream timeout reqid=%s", reqid)
        if not client_ws.closed:
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": "TTS request timed out"}
            )
    except (aiohttp.ClientError, OSError, RuntimeError) as err:
        logger.warning("[tts] stream failed reqid=%s: %s", reqid, err)
        if not client_ws.closed:
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": f"TTS failed: {err}"}
            )
    finally:
        if timeline_start_ms is not None:
            orchestrator = get_or_create_orchestrator(session_id)
            orchestrator.start()
            await orchestrator.record_event(
                TimelineEvent(
                    session_id=session_id,
                    modality="audio",
                    kind="tts_playback",
                    start_ms=timeline_start_ms,
                    end_ms=time.monotonic() * 1000,
                    payload={
                        "source": "system_output",
                        "request_id": reqid,
                        "generation_id": generation_id,
                        "status": timeline_status,
                        "text": timeline_text,
                    },
                    priority=30,
                )
            )
        if upstream_ws is not None and not upstream_ws.closed:
            await upstream_ws.close()
        if upstream_session is not None:
            await upstream_session.close()
        unregister_tts_generation(session_id, generation_id)


async def cancel_tts_stream_task(task):
    if task is None or task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=TTS_CANCEL_TIMEOUT)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.warning("[tts] previous stream did not stop within %.2fs", TTS_CANCEL_TIMEOUT)


async def cancel_tts_generation(
    session_id: str,
    generation_id: str | None = None,
) -> dict:
    active = _active_tts_generations.get(session_id)
    if active is None:
        return {
            "session_id": session_id,
            "generation_id": generation_id or "",
            "cancelled": False,
            "reason": "not_active",
        }
    active_generation_id, task = active
    if generation_id and generation_id != active_generation_id:
        return {
            "session_id": session_id,
            "generation_id": generation_id,
            "active_generation_id": active_generation_id,
            "cancelled": False,
            "reason": "generation_mismatch",
        }
    await cancel_tts_stream_task(task)
    unregister_tts_generation(session_id, active_generation_id)
    return {
        "session_id": session_id,
        "generation_id": active_generation_id,
        "cancelled": True,
    }


async def cleanup_tts_session(session_id: str) -> bool:
    result = await cancel_tts_generation(session_id)
    return bool(result["cancelled"])


async def record_tts_playback_progress(data: dict) -> None:
    session_id = str(data.get("session_id") or "web")
    now_ms = time.monotonic() * 1000
    try:
        played_audio_ms = max(0, float(data.get("played_audio_ms") or 0))
    except (TypeError, ValueError):
        played_audio_ms = 0
    await get_or_create_orchestrator(session_id).record_event(
        TimelineEvent(
            session_id=session_id,
            modality="audio",
            kind="tts_playback",
            start_ms=now_ms,
            end_ms=now_ms,
            payload={
                "source": "browser_output",
                "request_id": str(data.get("request_id") or ""),
                "generation_id": str(data.get("generation_id") or ""),
                "status": str(data.get("status") or "progress"),
                "played_audio_ms": played_audio_ms,
                "played_text": str(data.get("played_text") or ""),
                "text": str(data.get("text") or ""),
            },
            priority=35,
        )
    )


async def tts_websocket_handler(request):
    client_ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await client_ws.prepare(request)

    stream_task = None
    stream_generation_id = ""
    stream_session_id = ""

    try:
        async for msg in client_ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await client_ws.send_json({"type": "error", "error": "Invalid JSON"})
                    continue

                message_type = data.get("type") or "speak"
                if message_type == "speak":
                    await cancel_tts_stream_task(stream_task)
                    stream_session_id = str(data.get("session_id") or "web")
                    await cancel_tts_generation(stream_session_id)
                    stream_generation_id = str(data.get("generation_id") or "").strip()
                    if not stream_generation_id:
                        stream_generation_id = f"{stream_session_id}-{uuid.uuid4().hex}"
                        data["generation_id"] = stream_generation_id
                    stream_task = asyncio.create_task(run_tts_stream_request(client_ws, data))
                    register_tts_generation(
                        stream_session_id,
                        stream_generation_id,
                        stream_task,
                    )
                elif message_type == "stop":
                    requested_generation_id = str(data.get("generation_id") or "").strip()
                    result = await cancel_tts_generation(
                        str(data.get("session_id") or stream_session_id or "web"),
                        requested_generation_id or None,
                    )
                    if result["cancelled"]:
                        stream_task = None
                    await client_ws.send_json({"type": "stop_ack", **result})
                elif message_type == "playback_progress":
                    await record_tts_playback_progress(data)
                elif message_type == "ping":
                    await client_ws.send_json({"type": "pong", "id": data.get("id")})
                else:
                    await client_ws.send_json(
                        {"type": "error", "error": f"Unknown TTS message type: {message_type}"}
                    )
            elif msg.type == web.WSMsgType.ERROR:
                raise client_ws.exception() or RuntimeError("TTS client websocket error")
    except Exception as err:
        logger.warning("[tts] browser websocket failed: %s", err)
    finally:
        await cancel_tts_stream_task(stream_task)
        unregister_tts_generation(stream_session_id, stream_generation_id)
        if not client_ws.closed:
            await client_ws.close()

    return client_ws


async def tts_config_handler(request):
    return web.json_response(
        {
            "url": TTS_URL,
            "sample_rate": TTS_SAMPLE_RATE,
            "voice": TTS_VOICE,
            "emotion": TTS_EMOTION,
            "chunk_size": TTS_CHUNK_SIZE,
            "max_text_chars": TTS_MAX_TEXT_CHARS,
        }
    )


def setup_tts_routes(app):
    app.router.add_get("/api/tts/config", tts_config_handler)
    app.router.add_get("/api/tts", tts_websocket_handler)
