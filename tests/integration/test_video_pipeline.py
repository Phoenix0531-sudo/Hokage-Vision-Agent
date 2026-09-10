"""Video pipeline tests against a tiny synthetic MP4 generated with OpenCV."""

from pathlib import Path

import pytest

from hokage_vision.core.errors import VisionBackendError
from hokage_vision.vision.backends.mock import MockBackend
from hokage_vision.vision.inference import InferenceService

cv2 = pytest.importorskip("cv2")

FOURCC = "mp4v"
FPS = 10
WIDTH, HEIGHT = 160, 120
FRAME_COUNT = 12


def _write_video(path: Path) -> None:
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*FOURCC), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    for index in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), (19, 22, 31), dtype=np.uint8)
        # moving bright square so frames differ and the mock backend has structure
        x = 10 + index * 10
        frame[30:90, x : x + 60] = (90, 130, 200)
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="module")
def demo_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "demo.mp4"
    _write_video(path)
    return path


def test_detect_video_samples_frames_with_stride(demo_video: Path) -> None:
    service = InferenceService(MockBackend())

    summary = service.detect_video(demo_video, frame_stride=4)

    assert summary.frame_count == FRAME_COUNT
    # frames 0, 4, 8 are sampled
    assert summary.processed_frames == 3
    assert len(summary.detections_by_frame) == 3
    assert summary.metadata["frame_stride"] == 4
    assert summary.source == str(demo_video)


def test_detect_video_progress_callback_reports(demo_video: Path) -> None:
    service = InferenceService(MockBackend())
    progress: list[tuple[int, int]] = []

    summary = service.detect_video(
        demo_video, frame_stride=6, progress_callback=lambda i, t: progress.append((i, t))
    )

    assert summary.processed_frames == 2
    assert progress == [(1, FRAME_COUNT), (7, FRAME_COUNT)]


def test_detect_video_missing_file_raises(tmp_path: Path) -> None:
    service = InferenceService(MockBackend())

    with pytest.raises(VisionBackendError, match="Video could not be opened"):
        service.detect_video(tmp_path / "missing.mp4")
