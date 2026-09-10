"""Interaction-level GUI tests for the image detection panel widgets."""

from pathlib import Path

import pytest

from hokage_vision.core.types import DetectionResult
from hokage_vision.ui.widgets.image_detection_panel import ImageDetectionPanel
from hokage_vision.ui.widgets.result_table import ResultTable
from hokage_vision.ui.widgets.statistics_panel import StatisticsPanel
from hokage_vision.vision.backends.mock import MockBackend
from hokage_vision.vision.inference import InferenceService

pytestmark = pytest.mark.gui


def test_table_and_stats_update_after_detection(qtbot) -> None:
    panel = ImageDetectionPanel()
    qtbot.addWidget(panel)

    panel.detect_path(Path("examples/images/sample.jpg"))

    # three mock detections land in the table with real content
    assert panel.table.rowCount() == 3
    assert panel.table.item(0, 0).text() == "obito"
    assert panel.table.item(0, 1).text() == "0.91"

    # statistics label reflects class counts and average confidence
    text = panel.stats.summary.text()
    assert "obito" in text
    assert "Average confidence: 0.84" in text  # (0.91+0.84+0.77)/3

    # preview pixmaps actually rendered
    assert not panel.original.pixmap().isNull()
    assert not panel.result_image.pixmap().isNull()


def test_statistics_panel_empty_result(qtbot) -> None:
    stats = StatisticsPanel()
    qtbot.addWidget(stats)

    empty = DetectionResult(source="none", detections=[])
    stats.set_result(empty)

    assert stats.summary.text() == "Classes: {} | Average confidence: 0.00"


def test_result_table_clears_between_results(qtbot) -> None:
    table = ResultTable()
    qtbot.addWidget(table)
    service = InferenceService(MockBackend())

    full = service.detect_image(Path("examples/images/sample.jpg"))
    table.set_result(full)
    assert table.rowCount() == 3

    empty = DetectionResult(source="none", detections=[])
    table.set_result(empty)
    assert table.rowCount() == 0
