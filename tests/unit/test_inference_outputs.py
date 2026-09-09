from pathlib import Path

import pytest
from PIL import Image

from hokage_vision.vision.backends.mock import MockBackend
from hokage_vision.vision.inference import InferenceService


def _image(path: Path) -> Path:
    Image.new("RGB", (100, 100), (20, 30, 40)).save(path)
    return path


def test_detect_image_saves_rendered_and_json_outputs(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "input.png")
    service = InferenceService(MockBackend())
    out = tmp_path / "runs"

    result = service.detect_image(image_path, save_rendered=True, save_json=True, output_dir=out)

    assert result.rendered_image_path == out / "input_mock.jpg"
    assert (out / "input_mock.jpg").exists()
    assert result.metadata["json_path"].endswith("input.json")
    assert (out / "input.json").exists()
    assert '"backend": "mock"' in (out / "input.json").read_text(encoding="utf-8")


def test_detect_folder_missing_dir_raises_readable_error(tmp_path: Path) -> None:
    service = InferenceService(MockBackend())

    with pytest.raises(Exception, match="does not exist"):
        service.detect_folder(tmp_path / "missing")
