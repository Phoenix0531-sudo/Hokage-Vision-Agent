from pathlib import Path

from PIL import Image

from hokage_vision.data.split import list_images


def test_list_images_returns_supported_extensions_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "c.txt").write_text("not an image", encoding="utf-8")

    names = [path.name for path in list_images(tmp_path)]

    assert names == ["a.jpg", "b.png"]


def test_list_images_scans_nested_folders(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "inner"
    nested.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(nested / "deep.png")

    assert [path.name for path in list_images(tmp_path)] == ["deep.png"]


def test_list_images_empty_folder_returns_empty_list(tmp_path: Path) -> None:
    assert list_images(tmp_path) == []
