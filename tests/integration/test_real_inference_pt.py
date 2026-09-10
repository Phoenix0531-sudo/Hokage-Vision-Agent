"""Optional real .pt-weight tests: run only when local weights exist.

Unlike the ONNX integration tests (which require an exported model), these use
PyTorch weights directly. CI environments without `runs/closed-loop` weights
skip via the missing-file guard, so the suite stays green everywhere.
"""

from pathlib import Path

import pytest

from hokage_vision.core.errors import VisionBackendError
from hokage_vision.vision.backends.ultralytics_backend import UltralyticsBackend

ultralytics = pytest.importorskip("ultralytics")

PT_WEIGHTS = Path("runs/closed-loop/train/run/weights/best.pt")
PROBE_IMAGE = Path("runs/closed-loop/dataset/images/val/naruto_000.jpg")

pytestmark = pytest.mark.integration


def _backend() -> UltralyticsBackend:
    if not PT_WEIGHTS.exists():
        pytest.skip("no local .pt weights (run scripts/closed_loop_demo.py first)")
    backend = UltralyticsBackend(PT_WEIGHTS, conf_threshold=0.5)
    backend.load()
    return backend


def test_pt_backend_runs_real_inference() -> None:
    backend = _backend()

    result = backend.predict_image(PROBE_IMAGE)

    assert result.metadata["backend"] == "ultralytics"
    assert result.width == 320 and result.height == 240
    assert result.detections, "trained model should detect the synthetic naruto rectangle"
    assert result.detections[0].label == "naruto"
    for detection in result.detections:
        assert 0 <= detection.box.x1 <= detection.box.x2 <= result.width
        assert 0 <= detection.box.y1 <= detection.box.y2 <= result.height


def test_pt_backend_conf_threshold_filters() -> None:
    if not PT_WEIGHTS.exists():
        pytest.skip("no local .pt weights (run scripts/closed_loop_demo.py first)")

    backend = UltralyticsBackend(PT_WEIGHTS, conf_threshold=0.99)
    backend.load()

    result = backend.predict_image(PROBE_IMAGE)

    assert result.detections == []


def test_pt_backend_missing_model_raises(tmp_path: Path) -> None:
    backend = UltralyticsBackend(tmp_path / "nope.pt")

    with pytest.raises(VisionBackendError, match="Model file does not exist"):
        backend.load()
