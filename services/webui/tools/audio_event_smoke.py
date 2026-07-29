"""Classify one PCM16 WAV with the production CLAP audio-event backend."""

import argparse
import json
import wave

from joy_interaction_webui.omni.clap_audio_events import (
    ClapAudioEventBackend,
    ClapAudioEventConfig,
)
from joy_interaction_webui.omni.events import AudioWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--model", default="laion/clap-htsat-unfused")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with wave.open(args.wav, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        pcm = wav_file.readframes(frames)
    if channels != 1 or sample_width != 2:
        raise ValueError("WAV must be mono PCM16")

    duration_ms = frames / sample_rate * 1000
    window = AudioWindow(
        pcm=pcm,
        sample_rate=sample_rate,
        start_ms=0,
        end_ms=duration_ms,
        last_sequence=1,
        voice_ratio=0,
        latest_voice_active=False,
    )
    backend = ClapAudioEventBackend(
        ClapAudioEventConfig(model=args.model, device=args.device)
    )
    scores = sorted(backend.score(window), key=lambda item: item[1], reverse=True)
    print(
        json.dumps(
            {
                "wav": args.wav,
                "duration_seconds": round(duration_ms / 1000, 3),
                "scores": [
                    {"label": label, "confidence": round(confidence, 6)}
                    for label, confidence in scores
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
