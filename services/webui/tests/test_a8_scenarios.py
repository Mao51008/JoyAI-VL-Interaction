import pytest

from joy_interaction_webui.omni.decision import (
    ActionKind,
    DecisionPolicy,
    FakeResponseModel,
    OmniDecisionEngine,
    RuleBasedDecisionGate,
)
from joy_interaction_webui.omni.orchestrator import (
    MultimodalSnapshot,
    OmniOrchestrator,
    OrchestratorConfig,
)
from joy_interaction_webui.omni.timeline import TimelineEvent


def snapshot(
    sequence: int,
    events: list[TimelineEvent],
    *,
    trigger: str = "tick",
    end_ms: float = 10_000,
) -> MultimodalSnapshot:
    return MultimodalSnapshot(
        session_id="a8",
        sequence=sequence,
        trigger=trigger,
        trigger_event_kind="scenario" if trigger == "priority" else "",
        created_monotonic_ms=end_ms,
        window_start_ms=end_ms - 10_000,
        window_end_ms=end_ms,
        events=tuple(events),
    )


def engine(model: FakeResponseModel | None = None) -> OmniDecisionEngine:
    orchestrator = OmniOrchestrator("a8", clock=lambda: 10_000)
    return OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        model or FakeResponseModel(),
        clock=lambda: 10_000,
    )


@pytest.mark.asyncio
async def test_quiet_office_stays_silent() -> None:
    model = FakeResponseModel()
    decision_engine = engine(model)
    frame = TimelineEvent(
        "a8",
        "video",
        "frame",
        9000,
        9000,
        {"frame_index": 1, "width": 1280, "height": 720},
    )

    record = await decision_engine.evaluate_snapshot(snapshot(1, [frame]))

    assert record is not None
    assert record.action.kind == ActionKind.SILENCE
    assert model.calls == []


@pytest.mark.asyncio
async def test_visual_fire_without_audio_triggers_warning() -> None:
    decision_engine = engine()
    fire = TimelineEvent(
        "a8",
        "video",
        "visual_event",
        9000,
        9000,
        {"label": "fire", "confidence": 0.88},
        priority=100,
    )

    record = await decision_engine.evaluate_snapshot(snapshot(1, [fire], trigger="priority"))

    assert record is not None
    assert record.action.kind == ActionKind.RESPONSE
    assert record.trigger_reason == "dangerous_video:fire"
    assert record.confidence == pytest.approx(0.88)
    assert "危险" in record.action.text


@pytest.mark.asyncio
async def test_smoke_and_alarm_are_fused_into_one_urgent_decision() -> None:
    decision_engine = engine()
    smoke = TimelineEvent(
        "a8",
        "video",
        "visual_event",
        8800,
        8800,
        {"label": "smoke", "confidence": 0.81},
        priority=100,
    )
    alarm = TimelineEvent(
        "a8",
        "audio",
        "audio_event",
        9000,
        9000,
        {"label": "fire_alarm", "confidence": 0.93},
        priority=100,
    )

    record = await decision_engine.evaluate_snapshot(
        snapshot(1, [smoke, alarm], trigger="priority")
    )

    assert record is not None
    assert record.action.kind == ActionKind.RESPONSE
    assert record.trigger_reason == "multimodal_danger:smoke+fire_alarm"
    assert record.confidence == pytest.approx(0.93)


@pytest.mark.asyncio
async def test_changing_partial_speech_waits_for_final_before_responding() -> None:
    model = FakeResponseModel()
    decision_engine = engine(model)
    partial = TimelineEvent(
        "a8",
        "audio",
        "speech_partial",
        8500,
        9000,
        {"text": "帮我打开，不对"},
    )
    final = TimelineEvent(
        "a8",
        "audio",
        "speech_final",
        8500,
        9500,
        {"text": "请帮我关闭窗口", "confidence": 0.9},
    )

    partial_record = await decision_engine.evaluate_snapshot(snapshot(1, [partial]))
    final_record = await decision_engine.evaluate_snapshot(snapshot(2, [partial, final]))

    assert partial_record is not None
    assert partial_record.action.kind == ActionKind.SILENCE
    assert final_record is not None
    assert final_record.action.kind == ActionKind.RESPONSE
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_echo_with_new_user_content_preserves_barge_in() -> None:
    decision_engine = engine()
    playing = TimelineEvent(
        "a8",
        "audio",
        "tts_playback",
        8000,
        8000,
        {"status": "playing", "text": "请立即撤离", "generation_id": "generation-1"},
    )
    echo = TimelineEvent(
        "a8",
        "audio",
        "speech_partial",
        8500,
        8800,
        {
            "text": "请立即撤离",
            "metadata": {"likely_tts_echo": True, "echo_similarity": 1.0},
        },
    )
    user = TimelineEvent(
        "a8",
        "audio",
        "speech_partial",
        8500,
        9200,
        {
            "text": "不要说了，帮我报警",
            "metadata": {"echo_checked": True, "likely_tts_echo": False},
        },
    )

    record = await decision_engine.evaluate_snapshot(
        snapshot(1, [playing, echo, user], trigger="priority")
    )

    assert record is not None
    assert record.action.kind == ActionKind.INTERRUPT
    assert record.trigger_reason == "user_speech_during_tts"


@pytest.mark.asyncio
async def test_alarm_interrupts_old_tts_then_produces_emergency_response() -> None:
    decision_engine = engine()
    playing = TimelineEvent(
        "a8",
        "audio",
        "tts_playback",
        8000,
        8000,
        {"status": "playing", "text": "普通回答", "generation_id": "generation-1"},
    )
    alarm = TimelineEvent(
        "a8",
        "audio",
        "audio_event",
        9000,
        9000,
        {"label": "fire_alarm", "confidence": 0.94},
        priority=100,
    )

    interruption = await decision_engine.evaluate_snapshot(
        snapshot(1, [playing, alarm], trigger="priority")
    )
    cancelled = TimelineEvent(
        "a8",
        "audio",
        "tts_playback",
        8000,
        9200,
        {"status": "cancelled", "text": "普通回答", "generation_id": "generation-1"},
    )
    emergency = await decision_engine.evaluate_snapshot(
        snapshot(2, [playing, alarm, cancelled], trigger="priority")
    )

    assert interruption is not None
    assert interruption.action.kind == ActionKind.INTERRUPT
    assert emergency is not None
    assert emergency.action.kind == ActionKind.RESPONSE
    assert emergency.trigger_reason == "dangerous_audio:fire_alarm"
    assert "危险" in emergency.action.text


@pytest.mark.asyncio
async def test_overlapping_speech_is_reported_as_uncertain() -> None:
    decision_engine = engine()
    overlap = TimelineEvent(
        "a8",
        "audio",
        "speech_final",
        8500,
        9500,
        {
            "text": "重叠转写结果",
            "confidence": 0.42,
            "metadata": {"overlapping_speech": True},
        },
    )

    record = await decision_engine.evaluate_snapshot(snapshot(1, [overlap]))

    assert record is not None
    assert record.action.kind == ActionKind.RESPONSE
    assert record.trigger_reason == "speech_final_overlapping"
    assert record.confidence == pytest.approx(0.42)
    assert "可能不完整" in record.action.text


@pytest.mark.asyncio
async def test_virtual_30_minute_quiet_stream_remains_bounded() -> None:
    now_ms = [0.0]
    orchestrator = OmniOrchestrator(
        "long-stream",
        config=OrchestratorConfig(
            lookback_seconds=10,
            max_snapshot_history=64,
            timeline_max_age_seconds=120,
            timeline_max_events=300,
        ),
        clock=lambda: now_ms[0],
    )
    model = FakeResponseModel()
    decision_engine = OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        model,
        policy=DecisionPolicy(max_records=100),
        clock=lambda: now_ms[0],
    )

    for second in range(1800):
        now_ms[0] = second * 1000
        await orchestrator.record_event(
            TimelineEvent(
                "long-stream",
                "video",
                "frame",
                now_ms[0],
                now_ms[0],
                {"frame_index": second},
            )
        )
        current = await orchestrator.create_snapshot(trigger="tick")
        await decision_engine.evaluate_snapshot(current)

    assert len(orchestrator.timeline) <= 300
    assert len(orchestrator.snapshots) == 64
    assert len(decision_engine.records) == 100
    assert decision_engine.stats["decisions"] == 1800
    assert decision_engine.stats["responses"] == 0
    assert model.calls == []
