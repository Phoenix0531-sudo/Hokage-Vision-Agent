# Changelog

## Unreleased

### Added
- Closed-loop training demo: `scripts/closed_loop_demo.py` generates a synthetic dataset, fine-tunes yolov8n on CPU, exports ONNX, and runs real inference through `UltralyticsBackend` (24/24 val accuracy; see `models/model-card.synthetic-shapes.md`).
- Quickstart demo: `python examples/quickstart.py` walks detect → validate → smoke-train → agent → report in one command (see `docs/quickstart.md`).
- Real ONNX inference integration tests, video pipeline tests (synthetic MP4), 10 API endpoint tests, GUI interaction tests, and 38 new unit tests (39 → 86+ total).
- Demo video with real trained-model overlay: `examples/videos/demo.mp4`.
- Coverage gate (60%) with XML artifact upload in CI.

### Changed
- Removed vendored legacy YOLOv5 toolchain (`legacy/`, 177 files); provenance preserved in git history and documented in `docs/license-audit.md`, `docs/migration.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/README.md`.
- Fixed pre-existing lint warnings in `scripts/capture_real_shots.py` and `scripts/generate_evidence.py`; `ruff check` is now clean across the repo.
- Fixed environment-dependent assertion in `tests/unit/test_core_paths.py` that failed in Docker CI.

### Added (previous)
- Added Docker-first project structure and dependency-layer caching.
- Added mock backend, shared inference service, rendering, CLI, API, GUI, Agent tools, dataset validation, annotation assistance, smoke training, model registry, evaluation, and comparison foundations.
- Added package build, desktop build, CI, GUI tests, docs, release, CodeQL, Dependabot, and repository governance files.
- Added license audit, third-party notices, and data/model distribution guardrails.
