from pathlib import Path

from hokage_vision.reports.markdown import generate_markdown_report, summarize_detections


def test_generate_markdown_report_writes_sections(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "report.md"

    result = generate_markdown_report(
        "Test Report",
        output,
        summary={"files": 1},
        tool_results=[{"source": "a.png", "detections": []}],
    )

    assert result["status"] == "success"
    content = output.read_text(encoding="utf-8")
    assert content.startswith("# Test Report")
    assert "## Summary" in content
    assert "- files: 1" in content
    assert "## Tool Results" in content
    assert "## Data And License Notes" in content


def test_generate_markdown_report_without_summary_or_results(tmp_path: Path) -> None:
    output = tmp_path / "report.md"

    result = generate_markdown_report("Empty", output)

    assert result["status"] == "success"
    content = output.read_text(encoding="utf-8")
    assert "No additional summary was provided." in content
    assert "## Tool Results" not in content


def test_summarize_detections_counts_labels() -> None:
    results = [
        {"detections": [{"label": "obito"}, {"label": "naruto"}]},
        {"detections": [{"label": "obito"}]},
    ]

    summary = summarize_detections(results)

    assert summary == {"files": 2, "detections": 3, "classes": {"naruto": 1, "obito": 2}}


def test_summarize_detections_empty_results() -> None:
    assert summarize_detections([]) == {"files": 0, "detections": 0, "classes": {}}
