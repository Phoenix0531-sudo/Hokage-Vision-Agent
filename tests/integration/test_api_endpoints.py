from pathlib import Path

from fastapi.testclient import TestClient

from hokage_vision.api.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_detect_folder_mock_returns_results() -> None:
    client = _client()

    response = client.post("/detect/folder", json={"folder": "examples/images"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"], "expected at least one detection result"
    assert body["results"][0]["metadata"]["backend"] == "mock"


def test_detect_folder_missing_returns_400() -> None:
    client = _client()

    response = client.post("/detect/folder", json={"folder": "examples/does-not-exist"})

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_detect_image_unsupported_backend_returns_400() -> None:
    client = _client()

    response = client.post(
        "/detect/image",
        json={"image_path": "examples/images/sample.jpg", "backend": "unknown-net"},
    )

    assert response.status_code == 400
    assert "Unsupported backend" in response.json()["detail"]


def test_detect_image_conf_threshold_validation_returns_422() -> None:
    client = _client()

    response = client.post(
        "/detect/image",
        json={"image_path": "examples/images/sample.jpg", "conf_threshold": 2.0},
    )

    assert response.status_code == 422


def test_agent_run_detects_folder() -> None:
    client = _client()

    response = client.post(
        "/agent/run",
        json={"task": "批量识别 examples/images 文件夹里的目标"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"], "agent should have planned at least one tool call"
    assert body["tool_calls"][0]["name"] == "detect_folder"
    assert body["tool_calls"][0]["status"] == "success"


def test_agent_run_refuses_out_of_scope_task() -> None:
    client = _client()

    response = client.post("/agent/run", json={"task": "帮我写小说"})

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"] == []
    assert "Agent refused" in body["message"]


def test_dataset_validate_reports_example_dataset() -> None:
    client = _client()

    response = client.post(
        "/dataset/validate",
        json={"dataset_yaml": "configs/dataset.example.yaml"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["image_count"] > 0


def test_dataset_validate_missing_yaml_reports_issues() -> None:
    client = _client()

    response = client.post(
        "/dataset/validate",
        json={"dataset_yaml": "configs/definitely-missing.yaml"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["issues"]


def test_train_smoke_completes(tmp_path: Path) -> None:
    client = _client()

    response = client.post(
        "/train/smoke",
        json={"output_dir": str(tmp_path / "smoke"), "epochs": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["parameters"]["backend"] == "mock"


def test_models_compare_mock_rows(tmp_path: Path) -> None:
    client = _client()
    existing = tmp_path / "weights.pt"
    existing.write_bytes(b"fake")

    response = client.post(
        "/models/compare",
        json={"models": [str(existing), "missing.pt"], "mock": True},
    )

    assert response.status_code == 200
    rows = response.json()["models"]
    assert len(rows) == 2
    assert rows[0]["backend"] == "mock"
    assert rows[0]["size_bytes"] == existing.stat().st_size
    assert rows[1]["size_bytes"] is None
