from pathlib import Path

from hokage_vision.core.paths import project_root


def test_project_root_finds_pyproject_from_cwd() -> None:
    root = project_root()

    assert (root / "pyproject.toml").exists()
    assert root.name == "Hokage_Vision_Agent"


def test_project_root_falls_back_to_cwd_when_no_pyproject(tmp_path: Path, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    assert project_root(start=empty) == empty.resolve()


def test_project_root_walks_up_to_nearest_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    assert project_root(start=nested) == tmp_path.resolve()
