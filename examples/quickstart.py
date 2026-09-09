"""One-command quickstart demo: mock-first walkthrough of the core workflow.

Run from the repository root:

    python examples/quickstart.py

Every artifact is written under ``runs/quickstart/`` and uses the deterministic
mock backend, so the demo needs no GPU, no model weights, and no network.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hokage_vision.agents.providers.rule_based import RuleBasedAgent  # noqa: E402
from hokage_vision.data.validation import validate_yolo_dataset  # noqa: E402
from hokage_vision.reports.markdown import generate_markdown_report  # noqa: E402
from hokage_vision.training.smoke import run_smoke_training  # noqa: E402
from hokage_vision.vision.backends.mock import MockBackend  # noqa: E402
from hokage_vision.vision.inference import InferenceService  # noqa: E402

OUT = ROOT / "runs" / "quickstart"


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 58 - len(title)))


def main() -> None:
    sample = ROOT / "examples" / "images" / "sample.jpg"
    dataset_yaml = ROOT / "configs" / "dataset.example.yaml"

    banner("1. Detect one image (mock backend)")
    service = InferenceService(MockBackend())
    result = service.detect_image(sample, save_rendered=True, save_json=True, output_dir=OUT)
    print(f"image: {result.source}")
    for detection in result.detections:
        box = detection.box
        print(
            f"  {detection.label:<7} conf={detection.confidence:.2f} "
            f"box=({box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f})"
        )
    print(f"rendered: {result.rendered_image_path}")
    print(f"json: {result.metadata['json_path']}")

    banner("2. Detect a folder")
    results = service.detect_folder(ROOT / "examples" / "images")
    print(f"images processed: {len(results)}")

    banner("3. Validate the example YOLO dataset")
    report = validate_yolo_dataset(dataset_yaml)
    payload = asdict(report)
    print(f"valid: {report.valid} | images: {report.image_count} | boxes: {report.box_count}")
    for issue in payload["issues"]:
        print(f"  issue: {issue}")

    banner("4. Smoke training plan (safe by default)")
    job = run_smoke_training(output_dir=OUT / "smoke-train")
    print(f"status: {job['status']} | dry_run fields: {job['parameters']}")

    banner("5. Ask the rule-based agent")
    response = RuleBasedAgent().run(f"检测 {ROOT / 'examples' / 'images'} 里的图片")
    call = response.tool_calls[0]
    print(f"tool: {call.name} | status: {call.status} | count: {call.result['count']}")

    banner("6. Generate a markdown report")
    report_result = generate_markdown_report(
        "Quickstart Demo Report",
        OUT / "report.md",
        summary={"images": len(results), "backend": "mock"},
    )
    print(f"written: {report_result['path']}")

    banner("Done")
    print(f"All artifacts are under: {OUT}")


if __name__ == "__main__":
    main()
