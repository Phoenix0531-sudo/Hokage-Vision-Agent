from pathlib import Path

from hokage_vision.core.types import ModelInfo
from hokage_vision.training.registry import ModelRegistry


def test_list_models_returns_empty_when_registry_missing(tmp_path: Path) -> None:
    assert ModelRegistry(tmp_path / "missing.json").list_models() == []


def test_register_replaces_same_name_and_version(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register(ModelInfo("m", "0.1.0", None, "mock", ["obito"]))
    updated = registry.register(
        ModelInfo("m", "0.1.0", None, "ultralytics", ["obito"], notes="updated")
    )

    models = registry.list_models()

    assert len(models) == 1
    assert updated["backend"] == "ultralytics"
    assert models[0]["notes"] == "updated"


def test_register_keeps_different_versions_and_records_license(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register(ModelInfo("m", "0.1.0", None, "mock", ["obito"]))
    record = registry.register(
        ModelInfo("m", "0.2.0", Path("models/m.pt"), "ultralytics", ["obito"])
    )

    models = registry.list_models()

    assert len(models) == 2
    assert record["license"] == "TBD"
    assert record["created_at"].endswith("Z")
    assert models[1]["path"] == str(Path("models/m.pt"))
