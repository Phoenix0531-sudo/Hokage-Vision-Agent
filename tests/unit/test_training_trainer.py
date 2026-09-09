from pathlib import Path

import pytest
from PIL import Image

from hokage_vision.training.trainer import plan_yolo_training, run_yolo_training


def _make_dataset(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), (0, 0, 0)).save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/train\nnames: [obito]\nmanifest: manifest.yaml\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.yaml").write_text(
        "dataset:\n  name: tiny\n  redistribution_allowed: true\n"
        "sources:\n  - id: synthetic\n    path: images\n    redistribution_allowed: true\n"
        "classes: [obito]\n",
        encoding="utf-8",
    )
    return dataset_yaml


def test_plan_yolo_training_returns_dry_run_payload(tmp_path: Path) -> None:
    payload = plan_yolo_training(_make_dataset(tmp_path), epochs=3, batch=2)

    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert payload["parameters"]["epochs"] == 3
    assert payload["parameters"]["batch"] == 2
    assert payload["dataset_valid"] is True
    assert payload["dataset_issues"] == []


def test_run_yolo_training_defaults_to_dry_run(tmp_path: Path) -> None:
    payload = run_yolo_training(_make_dataset(tmp_path), dry_run=True)

    assert payload["dry_run"] is True
    assert payload["status"] == "planned"


def test_run_yolo_training_refuses_invalid_dataset_before_model_load(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "dataset.yaml"
    bad_yaml.write_text(
        "path: .\ntrain: images/missing\nval: images/missing\nnames: [obito]\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="validation failed"):
        run_yolo_training(bad_yaml, dry_run=False)
