"""Snapshot-driven non-turn-based decision engine used before a real VLM is attached."""

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol

from .orchestrator import MultimodalSnapshot, OmniOrchestrator
from .timeline import TimelineEvent


class ActionKind(StrEnum):
    SILENCE = "silence"
    RESPONSE = "response"
    DELEGATE = "delegate"
    INTERRUPT = "interrupt"


@dataclass(frozen=True)
class DecisionAction:
    kind: ActionKind
    text: str = ""
    raw: str = ""

    def protocol_text(self) -> str:
        marker = f"</{self.kind.value}>"
        return f"{marker} {self.text}".strip()


@dataclass(frozen=True)
class GateDecision:
    action: ActionKind
    reason: str
    urgent: bool = False
    confidence: float | None = None


@dataclass(frozen=True)
class DecisionContext:
    text: str
    fingerprint: str


@dataclass(frozen=True)
class DecisionRecord:
    snapshot_sequence: int
    snapshot_trigger: str
    trigger_reason: str
    context_fingerprint: str
    action: DecisionAction
    latency_ms: float
    confidence: float | None = None
    suppressed_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "snapshot_sequence": self.snapshot_sequence,
            "snapshot_trigger": self.snapshot_trigger,
            "trigger_reason": self.trigger_reason,
            "context_fingerprint": self.context_fingerprint,
            "action": self.action.kind.value,
            "text": self.action.text,
            "protocol_text": self.action.protocol_text(),
            "latency_ms": round(self.latency_ms, 3),
            "confidence": self.confidence,
            "suppressed_reason": self.suppressed_reason,
        }


@dataclass(frozen=True)
class DecisionPolicy:
    response_cooldown_seconds: float = 5.0
    duplicate_window_seconds: float = 30.0
    max_records: int = 500


class DecisionGate(Protocol):
    async def decide(
        self,
        snapshot: MultimodalSnapshot,
        context: DecisionContext,
    ) -> GateDecision: ...


class ResponseModel(Protocol):
    async def generate(
        self,
        snapshot: MultimodalSnapshot,
        context: DecisionContext,
        gate: GateDecision,
    ) -> str: ...


def parse_action(text: str) -> DecisionAction:
    raw = str(text or "").strip()
    markers = (
        ("</silence>", ActionKind.SILENCE),
        ("</response>", ActionKind.RESPONSE),
        ("</delegate>", ActionKind.DELEGATE),
        ("</delegation>", ActionKind.DELEGATE),
        ("</interrupt>", ActionKind.INTERRUPT),
    )
    for marker, kind in markers:
        if marker in raw:
            payload = raw.split(marker, 1)[1].strip()
            return DecisionAction(kind, payload, raw)
    return DecisionAction(ActionKind.RESPONSE, " ".join(raw.split()), raw)


class SnapshotContextBuilder:
    def __init__(self, *, max_chars: int = 12_000):
        self.max_chars = max_chars

    @staticmethod
    def _age(snapshot: MultimodalSnapshot, event: TimelineEvent) -> str:
        return f"-{max(0, snapshot.window_end_ms - event.end_ms) / 1000:.1f}s"

    def build(self, snapshot: MultimodalSnapshot) -> DecisionContext:
        groups: dict[str, list[str]] = {
            "Video observations": [],
            "Audio events": [],
            "Speech transcript": [],
            "User queries": [],
            "System speaking state": [],
            "Recent model actions": [],
        }
        for event in snapshot.events:
            age = self._age(snapshot, event)
            payload = event.payload
            if event.modality == "video" and event.kind == "frame":
                groups["Video observations"].append(
                    f"- {age} frame={payload.get('frame_index', '?')} "
                    f"size={payload.get('width', '?')}x{payload.get('height', '?')}"
                )
            elif event.kind == "audio_event":
                groups["Audio events"].append(
                    f"- {age} label={payload.get('label', '')} "
                    f"confidence={float(payload.get('confidence') or 0):.3f}"
                )
            elif event.kind in {"speech_partial", "speech_stable", "speech_final"}:
                text = str(payload.get("text") or payload.get("stable_prefix") or "")
                if text:
                    groups["Speech transcript"].append(f"- {age} {event.kind}: {text}")
            elif event.kind == "user_query":
                groups["User queries"].append(f"- {age} {(payload.get('text') or '')!s}")
            elif event.kind == "tts_playback":
                playback_text = payload.get("played_text") or payload.get("text") or ""
                groups["System speaking state"].append(
                    f"- {age} status={payload.get('status', '')} "
                    f"generation={payload.get('generation_id', '')} "
                    f"played_ms={float(payload.get('played_audio_ms') or 0):.0f} "
                    f"text={playback_text!s}"
                )
            elif event.modality == "model":
                groups["Recent model actions"].append(
                    f"- {age} {event.kind}: "
                    f"{(payload.get('action') or payload.get('text') or '')!s}"
                )

        parts = [
            f"Snapshot trigger: {snapshot.trigger}",
            f"Snapshot sequence: {snapshot.sequence}",
        ]
        for heading, lines in groups.items():
            parts.append(f"\n[{heading}]")
            parts.extend(lines[-20:] or ["- none"])
        text = "\n".join(parts)
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return DecisionContext(text=text, fingerprint=fingerprint)


class RuleBasedDecisionGate:
    """Deterministic first-stage gate suitable for fake-model development."""

    dangerous_labels: ClassVar[set[str]] = {
        "fire_alarm",
        "smoke_alarm",
        "explosion",
        "glass_break",
    }

    async def decide(
        self,
        snapshot: MultimodalSnapshot,
        context: DecisionContext,
    ) -> GateDecision:
        del context
        tts_events = [event for event in snapshot.events if event.kind == "tts_playback"]
        speech_starts = [event for event in snapshot.events if event.kind == "speech_start"]
        if tts_events and speech_starts:
            latest_tts = tts_events[-1]
            latest_speech = speech_starts[-1]
            if (
                latest_tts.payload.get("status") == "playing"
                and latest_speech.start_ms >= latest_tts.start_ms
            ):
                return GateDecision(
                    ActionKind.INTERRUPT,
                    "user_speech_during_tts",
                    urgent=True,
                )

        if tts_events and tts_events[-1].payload.get("status") == "playing":
            latest_tts = tts_events[-1]
            for event in reversed(snapshot.events):
                if (
                    event.kind == "audio_event"
                    and event.payload.get("label") in self.dangerous_labels
                    and event.start_ms >= latest_tts.start_ms
                ):
                    return GateDecision(
                        ActionKind.INTERRUPT,
                        f"dangerous_audio_during_tts:{event.payload.get('label')}",
                        urgent=True,
                        confidence=float(event.payload.get("confidence") or 0),
                    )

        for event in reversed(snapshot.events):
            if event.kind == "audio_event" and event.payload.get("label") in self.dangerous_labels:
                return GateDecision(
                    ActionKind.RESPONSE,
                    f"dangerous_audio:{event.payload.get('label')}",
                    urgent=True,
                    confidence=float(event.payload.get("confidence") or 0),
                )
        if any(event.kind == "user_query" for event in snapshot.events):
            return GateDecision(ActionKind.RESPONSE, "user_query")
        if any(event.kind == "speech_final" for event in snapshot.events):
            return GateDecision(ActionKind.RESPONSE, "speech_final")
        return GateDecision(ActionKind.SILENCE, "no_actionable_change")


class FakeResponseModel:
    """Scriptable no-GPU stand-in for the eventual JoyAI-VL decision call."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = deque(responses or [])
        self.calls: list[DecisionContext] = []

    async def generate(
        self,
        snapshot: MultimodalSnapshot,
        context: DecisionContext,
        gate: GateDecision,
    ) -> str:
        del snapshot, gate
        self.calls.append(context)
        if self.responses:
            return self.responses.popleft()
        return "</response> 已检测到需要关注的情况。"


DecisionCallback = Callable[[DecisionRecord], Awaitable[None] | None]


class OmniDecisionEngine:
    """Consume snapshots without queueing stale decisions and record every action."""

    def __init__(
        self,
        orchestrator: OmniOrchestrator,
        gate: DecisionGate,
        response_model: ResponseModel,
        *,
        context_builder: SnapshotContextBuilder | None = None,
        policy: DecisionPolicy | None = None,
        callback: DecisionCallback | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.orchestrator = orchestrator
        self.gate = gate
        self.response_model = response_model
        self.context_builder = context_builder or SnapshotContextBuilder()
        self.policy = policy or DecisionPolicy()
        self.callback = callback
        self.clock = clock or (lambda: time.monotonic() * 1000)
        self.records: deque[DecisionRecord] = deque(maxlen=self.policy.max_records)
        self._recent_responses: deque[tuple[float, str]] = deque()
        self._last_spoken_ms = float("-inf")
        self._queue: asyncio.Queue[MultimodalSnapshot] = asyncio.Queue(maxsize=1)
        self._worker: asyncio.Task | None = None
        self._last_snapshot_fingerprint = ""
        self.stats = {
            "received": 0,
            "decisions": 0,
            "silence": 0,
            "responses": 0,
            "interrupts": 0,
            "delegations": 0,
            "duplicates_suppressed": 0,
            "cooldown_suppressed": 0,
            "stale_snapshots_dropped": 0,
            "errors": 0,
        }

    def attach(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
            self.orchestrator.add_listener(self.enqueue_snapshot)

    def enqueue_snapshot(self, snapshot: MultimodalSnapshot) -> None:
        self.stats["received"] += 1
        if self._queue.full():
            queued = self._queue.get_nowait()
            self.stats["stale_snapshots_dropped"] += 1
            if snapshot.trigger != "priority" and queued.trigger == "priority":
                snapshot = queued
            self._queue.task_done()
        self._queue.put_nowait(snapshot)

    async def _run(self) -> None:
        while True:
            snapshot = await self._queue.get()
            try:
                await self.evaluate_snapshot(snapshot)
            except Exception:  # noqa: BLE001
                self.stats["errors"] += 1
            finally:
                self._queue.task_done()

    @staticmethod
    def _events_fingerprint(snapshot: MultimodalSnapshot) -> str:
        serialized = json.dumps(
            [event.to_dict() for event in snapshot.events],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _suppress(self, action: DecisionAction, gate: GateDecision, now_ms: float) -> str:
        if action.kind not in {ActionKind.RESPONSE, ActionKind.DELEGATE}:
            return ""
        normalized = re.sub(r"\s+", " ", action.text.casefold()).strip()
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        cutoff = now_ms - self.policy.duplicate_window_seconds * 1000
        while self._recent_responses and self._recent_responses[0][0] < cutoff:
            self._recent_responses.popleft()
        if normalized and any(item == fingerprint for _, item in self._recent_responses):
            self.stats["duplicates_suppressed"] += 1
            return "duplicate"
        if (
            not gate.urgent
            and now_ms - self._last_spoken_ms < self.policy.response_cooldown_seconds * 1000
        ):
            self.stats["cooldown_suppressed"] += 1
            return "cooldown"
        if normalized:
            self._recent_responses.append((now_ms, fingerprint))
        self._last_spoken_ms = now_ms
        return ""

    async def evaluate_snapshot(
        self,
        snapshot: MultimodalSnapshot,
    ) -> DecisionRecord | None:
        event_fingerprint = self._events_fingerprint(snapshot)
        if snapshot.trigger != "priority" and event_fingerprint == self._last_snapshot_fingerprint:
            self.stats["stale_snapshots_dropped"] += 1
            return None
        self._last_snapshot_fingerprint = event_fingerprint
        started = self.clock()
        context = self.context_builder.build(snapshot)
        gate = await self.gate.decide(snapshot, context)
        if gate.action in {ActionKind.SILENCE, ActionKind.INTERRUPT}:
            action = DecisionAction(gate.action)
        else:
            raw = await self.response_model.generate(snapshot, context, gate)
            action = parse_action(raw)

        now_ms = self.clock()
        suppressed = self._suppress(action, gate, now_ms)
        if suppressed:
            action = DecisionAction(ActionKind.SILENCE, raw=action.raw)
        record = DecisionRecord(
            snapshot_sequence=snapshot.sequence,
            snapshot_trigger=snapshot.trigger,
            trigger_reason=gate.reason,
            context_fingerprint=context.fingerprint,
            action=action,
            latency_ms=max(0, now_ms - started),
            confidence=gate.confidence,
            suppressed_reason=suppressed,
        )
        self.records.append(record)
        self.stats["decisions"] += 1
        stat_key = {
            ActionKind.SILENCE: "silence",
            ActionKind.RESPONSE: "responses",
            ActionKind.DELEGATE: "delegations",
            ActionKind.INTERRUPT: "interrupts",
        }[action.kind]
        self.stats[stat_key] += 1
        await self.orchestrator.record_event(
            TimelineEvent(
                session_id=self.orchestrator.session_id,
                modality="model",
                kind="decision",
                start_ms=now_ms,
                end_ms=now_ms,
                payload=record.to_dict(),
                priority=50 if action.kind != ActionKind.SILENCE else 5,
            )
        )
        if self.callback is not None:
            result = self.callback(record)
            if inspect.isawaitable(result):
                await result
        return record

    async def stop(self) -> None:
        self.orchestrator.remove_listener(self.enqueue_snapshot)
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def status(self) -> dict:
        return {
            "session_id": self.orchestrator.session_id,
            "running": self._worker is not None,
            "records": len(self.records),
            **self.stats,
        }


_decision_engines: dict[str, OmniDecisionEngine] = {}


def install_fake_decision_engine(
    orchestrator: OmniOrchestrator,
    *,
    callback: DecisionCallback | None = None,
    responses: list[str] | None = None,
    policy: DecisionPolicy | None = None,
) -> OmniDecisionEngine:
    existing = _decision_engines.get(orchestrator.session_id)
    if existing is not None:
        return existing
    engine = OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        FakeResponseModel(responses),
        policy=policy,
        callback=callback,
    )
    engine.attach()
    _decision_engines[orchestrator.session_id] = engine
    return engine


def get_decision_engine(session_id: str) -> OmniDecisionEngine | None:
    return _decision_engines.get(session_id)


async def cleanup_decision_engine(session_id: str) -> bool:
    engine = _decision_engines.pop(session_id, None)
    if engine is None:
        return False
    await engine.stop()
    return True
