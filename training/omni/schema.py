"""Versioned, source-traceable schema for time-aligned Omni training samples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "omni-training-v1"


class ActionKind(IntEnum):
    SILENCE = 0
    RESPONSE = 1
    DELEGATE = 2
    INTERRUPT = 3

    @classmethod
    def parse(cls, value: str) -> ActionKind:
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            allowed = ", ".join(member.name.lower() for member in cls)
            raise ValueError(
                f"unknown action {value!r}; expected one of: {allowed}"
            ) from exc


@dataclass(frozen=True)
class DataProvenance:
    dataset: str
    version: str
    source_uri: str
    license_name: str
    license_tier: str
    allows_training: bool
    allows_modification: bool
    allows_redistribution: bool
    allows_commercial_use: bool
    teacher_model: str = ""
    teacher_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DataProvenance:
        required = (
            "dataset",
            "version",
            "source_uri",
            "license_name",
            "license_tier",
            "allows_training",
            "allows_modification",
            "allows_redistribution",
            "allows_commercial_use",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"provenance missing fields: {', '.join(missing)}")
        tier = str(value["license_tier"])
        if tier not in {"redistributable", "research_only"}:
            raise ValueError("license_tier must be redistributable or research_only")
        if not bool(value["allows_training"]):
            raise ValueError("sample license does not allow training")
        return cls(
            dataset=str(value["dataset"]),
            version=str(value["version"]),
            source_uri=str(value["source_uri"]),
            license_name=str(value["license_name"]),
            license_tier=tier,
            allows_training=bool(value["allows_training"]),
            allows_modification=bool(value["allows_modification"]),
            allows_redistribution=bool(value["allows_redistribution"]),
            allows_commercial_use=bool(value["allows_commercial_use"]),
            teacher_model=str(value.get("teacher_model", "")),
            teacher_config=dict(value.get("teacher_config", {})),
        )


@dataclass(frozen=True)
class AudioSegment:
    path: str
    start_ms: int
    end_ms: int
    sample_rate: int
    num_samples: int
    channel: str = "user_audio"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AudioSegment:
        return cls(
            path=str(value.get("path", "")),
            start_ms=int(value["start_ms"]),
            end_ms=int(value["end_ms"]),
            sample_rate=int(value.get("sample_rate", 16_000)),
            num_samples=int(value["num_samples"]),
            channel=str(value.get("channel", "user_audio")),
        )

    def validate(self, duration_ms: int) -> None:
        _validate_range(self.start_ms, self.end_ms, duration_ms, "audio")
        if self.sample_rate <= 0 or self.num_samples <= 0:
            raise ValueError("audio sample_rate and num_samples must be positive")
        expected = (self.end_ms - self.start_ms) * self.sample_rate / 1000
        tolerance = max(self.sample_rate * 0.05, 1)
        if abs(self.num_samples - expected) > tolerance:
            raise ValueError(
                f"audio num_samples={self.num_samples} does not match "
                f"{self.end_ms - self.start_ms}ms at {self.sample_rate}Hz"
            )


@dataclass(frozen=True)
class VideoFrame:
    path: str
    timestamp_ms: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VideoFrame:
        return cls(path=str(value["path"]), timestamp_ms=int(value["timestamp_ms"]))


@dataclass(frozen=True)
class TextInput:
    text: str
    timestamp_ms: int
    channel: str = "user_text"
    auxiliary: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TextInput:
        return cls(
            text=str(value["text"]),
            timestamp_ms=int(value["timestamp_ms"]),
            channel=str(value.get("channel", "user_text")),
            auxiliary=bool(value.get("auxiliary", False)),
        )


@dataclass(frozen=True)
class DecisionTarget:
    timestamp_ms: int
    action: ActionKind
    text: str = ""
    semantic_tags: tuple[str, ...] = ()
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DecisionTarget:
        return cls(
            timestamp_ms=int(value["timestamp_ms"]),
            action=ActionKind.parse(str(value["action"])),
            text=str(value.get("text", "")),
            semantic_tags=tuple(str(tag) for tag in value.get("semantic_tags", [])),
            confidence=float(value.get("confidence", 1.0)),
        )

    def validate(self, duration_ms: int) -> None:
        if not 0 <= self.timestamp_ms < duration_ms:
            raise ValueError("target timestamp must be inside the sample duration")
        if self.action in {ActionKind.RESPONSE, ActionKind.DELEGATE} and not self.text:
            raise ValueError(f"{self.action.name.lower()} target requires text")
        if self.action in {ActionKind.SILENCE, ActionKind.INTERRUPT} and self.text:
            raise ValueError(f"{self.action.name.lower()} target must not include text")
        if not 0 <= self.confidence <= 1:
            raise ValueError("target confidence must be between 0 and 1")


@dataclass(frozen=True)
class OmniSample:
    sample_id: str
    duration_ms: int
    provenance: DataProvenance
    audio: tuple[AudioSegment, ...] = ()
    video: tuple[VideoFrame, ...] = ()
    text: tuple[TextInput, ...] = ()
    targets: tuple[DecisionTarget, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OmniSample:
        sample = cls(
            sample_id=str(value["sample_id"]),
            duration_ms=int(value["duration_ms"]),
            provenance=DataProvenance.from_dict(dict(value["provenance"])),
            audio=tuple(
                AudioSegment.from_dict(item) for item in value.get("audio", [])
            ),
            video=tuple(VideoFrame.from_dict(item) for item in value.get("video", [])),
            text=tuple(TextInput.from_dict(item) for item in value.get("text", [])),
            targets=tuple(
                DecisionTarget.from_dict(item) for item in value.get("targets", [])
            ),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value.get("schema_version", "")),
        )
        sample.validate()
        return sample

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if not (self.audio or self.video or self.text):
            raise ValueError("sample must contain at least one input modality")
        for segment in self.audio:
            segment.validate(self.duration_ms)
        for frame in self.video:
            if not 0 <= frame.timestamp_ms < self.duration_ms:
                raise ValueError("video timestamp must be inside the sample duration")
        for text in self.text:
            if not text.text.strip():
                raise ValueError("text input must not be empty")
            if not 0 <= text.timestamp_ms < self.duration_ms:
                raise ValueError("text timestamp must be inside the sample duration")
        for target in self.targets:
            target.validate(self.duration_ms)

    @property
    def modality_presence(self) -> tuple[bool, bool, bool]:
        return bool(self.audio), bool(self.video), bool(self.text)


def _validate_range(start_ms: int, end_ms: int, duration_ms: int, label: str) -> None:
    if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
        raise ValueError(
            f"{label} range [{start_ms}, {end_ms}) must be inside [0, {duration_ms})"
        )


def load_samples(path: Path) -> list[OmniSample]:
    """Load JSONL samples and report the failing line number."""
    samples: list[OmniSample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(OmniSample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"{path}: no samples found")
    return samples
