from pathlib import Path

import pytest

from hokage_vision.data.yolo_dataset import class_names, load_yolo_dataset_yaml


def test_load_yolo_dataset_yaml_returns_mapping(tmp_path: Path) -> None:
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nnames: [obito, naruto]\n", encoding="utf-8")

    data = load_yolo_dataset_yaml(yaml_path)

    assert data["train"] == "images/train"
    assert data["names"] == ["obito", "naruto"]


def test_class_names_accepts_list_and_sorted_dict() -> None:
    assert class_names({"names": ["obito", "naruto"]}) == ["obito", "naruto"]
    assert class_names({"names": {1: "naruto", 0: "obito"}}) == ["obito", "naruto"]


def test_load_yolo_dataset_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_yolo_dataset_yaml(yaml_path)


def test_load_yolo_dataset_yaml_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("", encoding="utf-8")

    assert load_yolo_dataset_yaml(yaml_path) == {}
