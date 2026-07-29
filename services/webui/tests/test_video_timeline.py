import numpy as np
import pytest

from joy_interaction_webui.video_processor import VideoProcessorTrack


@pytest.mark.asyncio
async def test_video_processor_reports_sampled_frame_on_monotonic_timeline() -> None:
    recorded = []

    async def collect(metadata):
        recorded.append(metadata)

    processor = VideoProcessorTrack(
        track=None,
        vlm_service=None,
        timeline_callback=collect,
    )
    processor.frame_count = 7

    await processor._record_timeline_frame(
        {
            "timestamp": 1.5,
            "timestamp_kind": "relative_seconds",
            "pts": 135000,
            "timestamp_interval_seconds": 1.0,
        },
        frame_shape=np.zeros((360, 640, 3), dtype=np.uint8).shape,
        frames_per_batch=2,
    )

    assert len(recorded) == 1
    assert recorded[0]["frame_index"] == 7
    assert recorded[0]["height"] == 360
    assert recorded[0]["width"] == 640
    assert recorded[0]["frames_per_batch"] == 2
    assert recorded[0]["server_monotonic_ms"] > 0
