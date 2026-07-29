"""Session-scoped multimodal timeline and snapshot scheduler."""

import asyncio
import inspect
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .timeline import TimelineBuffer, TimelineEvent

SnapshotListener = Callable[["MultimodalSnapshot"], Awaitable[None] | None]


@dataclass(frozen=True)
class OrchestratorConfig:
    tick_seconds: float = 1.0
    lookback_seconds: float = 10.0
    urgent_priority: int = 80
    max_snapshot_history: int = 256
    timeline_max_age_seconds: float = 120.0
    timeline_max_events: int = 2000


@dataclass(frozen=True)
class MultimodalSnapshot:
    session_id: str
    sequence: int
    trigger: str
    created_monotonic_ms: float
    window_start_ms: float
    window_end_ms: float
    events: tuple[TimelineEvent, ...]
    trigger_event_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "trigger": self.trigger,
            "trigger_event_kind": self.trigger_event_kind,
            "created_monotonic_ms": self.created_monotonic_ms,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "events": [event.to_dict() for event in self.events],
        }


class OmniOrchestrator:
    def __init__(
        self,
        session_id: str,
        *,
        config: OrchestratorConfig | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.session_id = session_id
        self.config = config or OrchestratorConfig()
        self.clock = clock or (lambda: time.monotonic() * 1000)
        self.timeline = TimelineBuffer(
            max_age_seconds=self.config.timeline_max_age_seconds,
            max_events=self.config.timeline_max_events,
        )
        self.snapshots: deque[MultimodalSnapshot] = deque(
            maxlen=self.config.max_snapshot_history
        )
        self.listeners: set[SnapshotListener] = set()
        self.running = False
        self._task: asyncio.Task | None = None
        self._sequence = 0
        self.stats = {
            "events": 0,
            "ticks": 0,
            "priority_triggers": 0,
            "snapshots": 0,
            "listener_errors": 0,
        }

    def add_listener(self, listener: SnapshotListener) -> None:
        self.listeners.add(listener)

    def remove_listener(self, listener: SnapshotListener) -> None:
        self.listeners.discard(listener)

    async def record_event(self, event: TimelineEvent) -> MultimodalSnapshot | None:
        if event.session_id != self.session_id:
            raise ValueError("timeline event session_id does not match orchestrator")
        self.timeline.append(event)
        self.stats["events"] += 1
        if event.priority >= self.config.urgent_priority:
            self.stats["priority_triggers"] += 1
            return await self.create_snapshot(
                trigger="priority",
                trigger_event_kind=event.kind,
            )
        return None

    async def create_snapshot(
        self,
        *,
        trigger: str,
        trigger_event_kind: str = "",
    ) -> MultimodalSnapshot:
        now_ms = max(self.clock(), self.timeline.latest_ms)
        self._sequence += 1
        snapshot = MultimodalSnapshot(
            session_id=self.session_id,
            sequence=self._sequence,
            trigger=trigger,
            trigger_event_kind=trigger_event_kind,
            created_monotonic_ms=now_ms,
            window_start_ms=now_ms - self.config.lookback_seconds * 1000,
            window_end_ms=now_ms,
            events=self.timeline.snapshot(
                now_ms=now_ms,
                lookback_seconds=self.config.lookback_seconds,
            ),
        )
        self.snapshots.append(snapshot)
        self.stats["snapshots"] += 1
        for listener in tuple(self.listeners):
            try:
                result = listener(snapshot)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                self.stats["listener_errors"] += 1
        return snapshot

    async def _run(self) -> None:
        while self.running:
            await asyncio.sleep(self.config.tick_seconds)
            if not self.running:
                break
            self.stats["ticks"] += 1
            await self.create_snapshot(trigger="tick")

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "running": self.running,
            "timeline_events": len(self.timeline),
            "snapshot_history": len(self.snapshots),
            **self.stats,
        }


def orchestrator_config_from_env() -> OrchestratorConfig:
    return OrchestratorConfig(
        tick_seconds=float(os.getenv("OMNI_TICK_SECONDS", "1")),
        lookback_seconds=float(os.getenv("OMNI_LOOKBACK_SECONDS", "10")),
        urgent_priority=int(os.getenv("OMNI_URGENT_PRIORITY", "80")),
        max_snapshot_history=int(os.getenv("OMNI_MAX_SNAPSHOTS", "256")),
        timeline_max_age_seconds=float(os.getenv("OMNI_TIMELINE_SECONDS", "120")),
        timeline_max_events=int(os.getenv("OMNI_TIMELINE_MAX_EVENTS", "2000")),
    )


_orchestrators: dict[str, OmniOrchestrator] = {}


def get_or_create_orchestrator(session_id: str) -> OmniOrchestrator:
    orchestrator = _orchestrators.get(session_id)
    if orchestrator is None:
        orchestrator = OmniOrchestrator(
            session_id,
            config=orchestrator_config_from_env(),
        )
        _orchestrators[session_id] = orchestrator
    return orchestrator


def get_orchestrator(session_id: str) -> OmniOrchestrator | None:
    return _orchestrators.get(session_id)


async def cleanup_orchestrator(session_id: str) -> bool:
    orchestrator = _orchestrators.pop(session_id, None)
    if orchestrator is None:
        return False
    await orchestrator.stop()
    return True


async def clear_orchestrators() -> None:
    orchestrators = list(_orchestrators.values())
    _orchestrators.clear()
    for orchestrator in orchestrators:
        await orchestrator.stop()
