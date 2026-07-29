"""Continuous multimodal input primitives used by the WebUI."""

from .decision import (
    ActionKind,
    DecisionAction,
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
from .events import AudioTimelineEvent
from .orchestrator import MultimodalSnapshot, OmniOrchestrator, OrchestratorConfig
from .streaming_asr import (
    StreamingASRConfig,
    StreamingASRCoordinator,
    TranscriptionResult,
    WindowTranscriber,
)
from .timeline import TimelineBuffer, TimelineEvent
from .vllm_asr import VllmASRConfig, VllmWindowTranscriber

__all__ = [
    "ActionKind",
    "AudioTimelineEvent",
    "DecisionAction",
    "DecisionPolicy",
    "FakeResponseModel",
    "MultimodalSnapshot",
    "OmniDecisionEngine",
    "OmniOrchestrator",
    "OrchestratorConfig",
    "RuleBasedDecisionGate",
    "SnapshotContextBuilder",
    "StreamingASRConfig",
    "StreamingASRCoordinator",
    "TimelineBuffer",
    "TimelineEvent",
    "TranscriptionResult",
    "VllmASRConfig",
    "VllmWindowTranscriber",
    "WindowTranscriber",
    "cleanup_decision_engine",
    "get_decision_engine",
    "install_fake_decision_engine",
    "parse_action",
]
