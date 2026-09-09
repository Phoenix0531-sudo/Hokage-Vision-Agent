from pathlib import Path

from hokage_vision.training.smoke import run_smoke_training


def test_run_smoke_training_reports_completed_mock_job(tmp_path: Path) -> None:
    result = run_smoke_training(output_dir=tmp_path / "smoke", epochs=2)

    assert result["name"] == "hokage-yolo-smoke"
    assert result["status"] == "completed"
    assert result["dry_run"] is False
    assert result["parameters"]["epochs"] == 2
    assert result["parameters"]["backend"] == "mock"
    assert result["metrics"]["map50"] is None
    assert "no real model weights" in result["message"]
