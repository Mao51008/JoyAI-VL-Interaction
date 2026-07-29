"""Feed a PCM16 WAV through the continuous-audio WebSocket."""

import argparse
import asyncio
import json
import math
import struct
import time
import uuid
import wave

import aiohttp

AUDIO_PACKET_HEADER = struct.Struct(">4sBBHIdII")
PACKET_SAMPLES = 640


def packet_bytes(
    pcm: bytes,
    *,
    session_id: str,
    sequence: int,
    timestamp_ms: float,
    voice_active: bool,
) -> bytes:
    session = session_id.encode("utf-8")
    header = AUDIO_PACKET_HEADER.pack(
        b"JAI1",
        1,
        1 if voice_active else 0,
        len(session),
        sequence,
        timestamp_ms,
        16000,
        len(pcm) // 2,
    )
    return header + session + pcm


def rms(pcm: bytes) -> float:
    samples = memoryview(pcm).cast("h")
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768


async def receive_events(ws, events: list[dict]) -> None:
    async for message in ws:
        if message.type != aiohttp.WSMsgType.TEXT:
            continue
        payload = json.loads(message.data)
        if payload.get("type") == "audio_event":
            events.append(payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)


async def run(args) -> int:
    with wave.open(args.wav, "rb") as wav_file:
        if (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
        ) != (1, 2, 16000):
            raise ValueError("WAV must be mono, PCM16, 16000 Hz")
        pcm = wav_file.readframes(wav_file.getnframes()) * args.repeat

    session_id = args.session_id or f"smoke-{uuid.uuid4().hex[:8]}"
    separator = "&" if "?" in args.url else "?"
    url = f"{args.url}{separator}session_id={session_id}"
    events: list[dict] = []
    async with (
        aiohttp.ClientSession() as client,
        client.ws_connect(url, ssl=False, heartbeat=20) as ws,
    ):
        status = await ws.receive_json()
        print(json.dumps(status, ensure_ascii=False), flush=True)
        receiver = asyncio.create_task(receive_events(ws, events))
        sequence = 1
        started_ms = time.monotonic() * 1000
        for offset in range(0, len(pcm), PACKET_SAMPLES * 2):
            chunk = pcm[offset : offset + PACKET_SAMPLES * 2]
            if len(chunk) < PACKET_SAMPLES * 2:
                chunk += b"\0" * (PACKET_SAMPLES * 2 - len(chunk))
            await ws.send_bytes(
                packet_bytes(
                    chunk,
                    session_id=session_id,
                    sequence=sequence,
                    timestamp_ms=started_ms + (sequence - 1) * 40,
                    voice_active=rms(chunk) >= args.vad_threshold,
                )
            )
            sequence += 1
            await asyncio.sleep(0.04 / args.speed)

        for _ in range(25):
            await ws.send_bytes(
                packet_bytes(
                    b"\0\0" * PACKET_SAMPLES,
                    session_id=session_id,
                    sequence=sequence,
                    timestamp_ms=started_ms + (sequence - 1) * 40,
                    voice_active=False,
                )
            )
            sequence += 1
            await asyncio.sleep(0.04 / args.speed)

        await asyncio.sleep(args.final_wait)
        await ws.send_json({"type": "end"})
        await ws.close()
        await receiver

    kinds = [event.get("kind") for event in events]
    if "speech_partial" not in kinds or "speech_end" not in kinds:
        raise RuntimeError(f"incomplete event flow: {kinds}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--url", default="ws://127.0.0.1:8099/ws/audio-ingress")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--vad-threshold", type=float, default=0.008)
    parser.add_argument("--final-wait", type=float, default=2.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
