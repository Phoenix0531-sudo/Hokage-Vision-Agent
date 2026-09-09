from hokage_vision.core.i18n import translate


def test_translate_known_key_in_zh() -> None:
    assert translate("backend.mock", "zh-CN") == "Mock 后端"


def test_translate_known_key_in_en() -> None:
    assert translate("backend.mock", "en-US") == "Mock backend"


def test_translate_unknown_language_falls_back_to_key() -> None:
    assert translate("backend.mock", "fr-FR") == "backend.mock"


def test_translate_unknown_key_returns_key() -> None:
    assert translate("missing.key", "zh-CN") == "missing.key"
