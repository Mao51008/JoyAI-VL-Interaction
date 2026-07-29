import pytest

from joy_interaction_webui.omni.decision import (
    ActionKind,
    DecisionPolicy,
    FakeResponseModel,
    OmniDecisionEngine,
    RuleBasedDecisionGate,
    SnapshotContextBuilder,
    cleanup_decision_engine,
    get_decision_engine,
    install_fake_decision_engine,
    parse_action,
)
from joy_interaction_webui.omni.orchestrator import (
    MultimodalSnapshot,
    OmniOrchestrator,
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
        session_id="s",
        sequence=sequence,
        trigger=trigger,
        trigger_event_kind="audio_event" if trigger == "priority" else "",
        created_monotonic_ms=end_ms,
        window_start_ms=end_ms - 10_000,
        window_end_ms=end_ms,
        events=tuple(events),
    )


def test_action_protocol_parser_supports_all_actions_and_plain_text() -> None:
    assert parse_action("</silence>").kind == ActionKind.SILENCE
    assert parse_action("</response> 请立即撤离").text == "请立即撤离"
    assert parse_action("</delegate> 检查消防预案").kind == ActionKind.DELEGATE
    assert parse_action("</delegation> 检查消防预案").kind == ActionKind.DELEGATE
    assert parse_action("</interrupt>").kind == ActionKind.INTERRUPT
    assert parse_action("普通回答").protocol_text() == "</response> 普通回答"


def test_context_builder_formats_all_modalities() -> None:
    events = [
        TimelineEvent("s", "video", "frame", 1000, 1000, {"frame_index": 3}),
        TimelineEvent(
            "s",
            "audio",
            "audio_event",
            2000,
            2100,
            {"label": "smoke_alarm", "confidence": 0.91},
        ),
        TimelineEvent("s", "audio", "speech_final", 2200, 2300, {"text": "快出去"}),
        TimelineEvent("s", "text", "user_query", 2400, 2400, {"text": "怎么了"}),
        TimelineEvent(
            "s",
            "audio",
            "tts_playback",
            2500,
            2600,
            {"status": "done", "text": "请撤离"},
        ),
        TimelineEvent("s", "model", "model_action", 2700, 2700, {"action": "response"}),
    ]

    context = SnapshotContextBuilder().build(snapshot(1, events))

    assert "[Video observations]" in context.text
    assert "smoke_alarm" in context.text
    assert "speech_final: 快出去" in context.text
    assert "怎么了" in context.text
    assert "status=done" in context.text
    assert "model_action" in context.text
    assert len(context.fingerprint) == 16


@pytest.mark.asyncio
async def test_normal_scene_stays_silent_without_calling_fake_vlm() -> None:
    orchestrator = OmniOrchestrator("s", clock=lambda: 10_000)
    model = FakeResponseModel()
    engine = OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        model,
        clock=lambda: 10_000,
    )

    record = await engine.evaluate_snapshot(
        snapshot(
            1,
            [TimelineEvent("s", "video", "frame", 9000, 9000, {"frame_index": 1})],
        )
    )

    assert record is not None
    assert record.action.kind == ActionKind.SILENCE
    assert record.trigger_reason == "no_actionable_change"
    assert model.calls == []


@pytest.mark.asyncio
async def test_fire_alarm_responds_but_duplicate_warning_is_suppressed() -> None:
    now = [10_000.0]
    orchestrator = OmniOrchestrator("s", clock=lambda: now[0])
    model = FakeResponseModel(
        [
            "</response> 检测到火警，请立即撤离。",
            "</response> 检测到火警，请立即撤离。",
            "</response> 检测到爆炸声，请远离危险区域。",
        ]
    )
    engine = OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        model,
        policy=DecisionPolicy(
            response_cooldown_seconds=10,
            duplicate_window_seconds=30,
        ),
        clock=lambda: now[0],
    )
    fire = TimelineEvent(
        "s",
        "audio",
        "audio_event",
        9000,
        9000,
        {"label": "fire_alarm", "confidence": 0.9},
        priority=100,
    )

    first = await engine.evaluate_snapshot(snapshot(1, [fire], trigger="priority"))
    now[0] += 1000
    duplicate = await engine.evaluate_snapshot(snapshot(2, [fire], trigger="priority"))
    now[0] += 1000
    explosion = TimelineEvent(
        "s",
        "audio",
        "audio_event",
        10_500,
        10_500,
        {"label": "explosion", "confidence": 0.8},
        priority=100,
    )
    different = await engine.evaluate_snapshot(
        snapshot(3, [fire, explosion], trigger="priority", end_ms=12_000)
    )

    assert first is not None and first.action.kind == ActionKind.RESPONSE
    assert first.confidence == pytest.approx(0.9)
    assert duplicate is not None and duplicate.action.kind == ActionKind.SILENCE
    assert duplicate.suppressed_reason == "duplicate"
    assert different is not None and different.action.kind == ActionKind.RESPONSE
    assert engine.stats["duplicates_suppressed"] == 1
    assert engine.stats["cooldown_suppressed"] == 0


@pytest.mark.asyncio
async def test_nonurgent_responses_obey_cooldown() -> None:
    now = [10_000.0]
    orchestrator = OmniOrchestrator("s", clock=lambda: now[0])
    model = FakeResponseModel(["</response> 第一个回答", "</response> 第二个不同的回答"])
    engine = OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        model,
        policy=DecisionPolicy(response_cooldown_seconds=5),
        clock=lambda: now[0],
    )
    query1 = TimelineEvent("s", "text", "user_query", 9000, 9000, {"text": "问题一"})
    query2 = TimelineEvent("s", "text", "user_query", 10_500, 10_500, {"text": "问题二"})

    first = await engine.evaluate_snapshot(snapshot(1, [query1]))
    now[0] += 1000
    second = await engine.evaluate_snapshot(snapshot(2, [query1, query2], end_ms=11_000))

    assert first is not None and first.action.kind == ActionKind.RESPONSE
    assert second is not None and second.action.kind == ActionKind.SILENCE
    assert second.suppressed_reason == "cooldown"


@pytest.mark.asyncio
async def test_user_speech_during_tts_produces_interrupt_without_response_call() -> None:
    orchestrator = OmniOrchestrator("s", clock=lambda: 10_000)
    model = FakeResponseModel()
    engine = OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        model,
        clock=lambda: 10_000,
    )
    playing = TimelineEvent(
        "s",
        "audio",
        "tts_playback",
        8000,
        8000,
        {"status": "playing", "source": "system_output"},
    )
    speech = TimelineEvent("s", "audio", "speech_start", 9000, 9000)

    record = await engine.evaluate_snapshot(snapshot(1, [playing, speech], trigger="priority"))

    assert record is not None
    assert record.action.kind == ActionKind.INTERRUPT
    assert record.trigger_reason == "user_speech_during_tts"
    assert model.calls == []
    assert engine.stats["interrupts"] == 1


@pytest.mark.asyncio
async def test_decision_record_is_written_back_to_timeline() -> None:
    orchestrator = OmniOrchestrator("s", clock=lambda: 10_000)
    engine = OmniDecisionEngine(
        orchestrator,
        RuleBasedDecisionGate(),
        FakeResponseModel(["</delegate> 交给后台检查"]),
        clock=lambda: 10_000,
    )
    query = TimelineEvent("s", "text", "user_query", 9000, 9000, {"text": "深入分析"})

    record = await engine.evaluate_snapshot(snapshot(1, [query]))

    assert record is not None
    assert record.action.kind == ActionKind.DELEGATE
    decision_event = orchestrator.timeline.snapshot()[-1]
    assert decision_event.kind == "decision"
    assert decision_event.payload["trigger_reason"] == "user_query"
    assert decision_event.payload["protocol_text"] == "</delegate> 交给后台检查"


@pytest.mark.asyncio
async def test_fake_engine_registry_reuses_and_cleans_up_session_engine() -> None:
    orchestrator = OmniOrchestrator("registry")

    first = install_fake_decision_engine(orchestrator)
    second = install_fake_decision_engine(orchestrator)

    assert first is second
    assert get_decision_engine("registry") is first
    assert first.status()["running"] is True

    assert await cleanup_decision_engine("registry") is True
    assert get_decision_engine("registry") is None
    assert first.status()["running"] is False
    assert await cleanup_decision_engine("registry") is False
