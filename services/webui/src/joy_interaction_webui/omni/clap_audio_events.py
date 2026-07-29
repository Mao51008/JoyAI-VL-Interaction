"""Zero-shot environmental sound detection with CLAP."""

import asyncio
import threading
from dataclasses import dataclass

import numpy as np

from .audio_events import AudioEventDetection
from .events import AudioWindow

DEFAULT_LABEL_PROMPTS = (
    ("fire_alarm", "the sound of a fire alarm"),
    ("smoke_alarm", "the sound of a smoke detector alarm"),
    ("glass_break", "the sound of glass breaking"),
    ("explosion", "the sound of an explosion"),
    ("background", "ordinary quiet indoor background sound"),
)


@dataclass(frozen=True)
class ClapAudioEventConfig:
    model: str = "laion/clap-htsat-unfused"
    device: str = "cpu"
    confidence_threshold: float = 0.55
    cooldown_seconds: float = 2.0
    label_prompts: tuple[tuple[str, str], ...] = DEFAULT_LABEL_PROMPTS


class ClapAudioEventBackend:
    """Lazily load one model and serialize inference shared by all sessions."""

    def __init__(self, config: ClapAudioEventConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import ClapModel, ClapProcessor

        self._processor = ClapProcessor.from_pretrained(self.config.model)
        self._model = ClapModel.from_pretrained(self.config.model).to(self.config.device)
        self._model.eval()

    def score(self, window: AudioWindow) -> list[tuple[str, float]]:
        import torch

        waveform = np.frombuffer(window.pcm, dtype="<i2").astype(np.float32) / 32768.0
        prompts = [prompt for _, prompt in self.config.label_prompts]
        with self._lock, torch.inference_mode():
            self._load()
            processor_args = {
                "text": prompts,
                "audio": [waveform],
                "sampling_rate": window.sample_rate,
                "return_tensors": "pt",
                "padding": True,
            }
            try:
                inputs = self._processor(**processor_args)
            except (TypeError, ValueError) as err:
                if "audio" not in str(err):
                    raise
                processor_args["audios"] = processor_args.pop("audio")
                inputs = self._processor(**processor_args)
            inputs = {name: value.to(self.config.device) for name, value in inputs.items()}
            scores = self._model(**inputs).logits_per_audio.softmax(dim=-1)[0]
            confidences = scores.detach().cpu().tolist()
        return [
            (label, float(confidence))
            for (label, _), confidence in zip(
                self.config.label_prompts, confidences, strict=True
            )
        ]

    def classify(self, window: AudioWindow) -> tuple[str, float]:
        return max(self.score(window), key=lambda item: item[1])


class ClapAudioEventDetector:
    def __init__(
        self,
        config: ClapAudioEventConfig,
        backend: ClapAudioEventBackend,
    ):
        self.config = config
        self.backend = backend
        self._last_emitted_ms: dict[str, float] = {}

    async def detect(self, window: AudioWindow) -> list[AudioEventDetection]:
        label, confidence = await asyncio.to_thread(self.backend.classify, window)
        if label == "background" or confidence < self.config.confidence_threshold:
            return []
        last_emitted = self._last_emitted_ms.get(label, float("-inf"))
        if window.end_ms - last_emitted < self.config.cooldown_seconds * 1000:
            return []
        self._last_emitted_ms[label] = window.end_ms
        return [
            AudioEventDetection(
                label=label,
                confidence=confidence,
                start_ms=window.start_ms,
                end_ms=window.end_ms,
            )
        ]
