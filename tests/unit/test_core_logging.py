from hokage_vision.core.logging import get_logger


def test_get_logger_returns_configured_logger() -> None:
    logger = get_logger("hokage_vision_test_logger")

    assert logger.name == "hokage_vision_test_logger"
    assert logger.level == 20  # INFO
    assert len(logger.handlers) == 1
    assert logger.handlers[0].formatter is not None


def test_get_logger_is_idempotent_per_name() -> None:
    first = get_logger("hokage_vision_test_idempotent")
    second = get_logger("hokage_vision_test_idempotent")

    assert first is second
    assert len(first.handlers) == 1
