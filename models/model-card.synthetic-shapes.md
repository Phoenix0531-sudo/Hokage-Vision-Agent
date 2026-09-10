# Model Card: hokage-synthetic-shapes-yolov8n

## Release Status

- Status: `reviewed-external`
- Version: `0.1.0`
- Registry entry: `models/registry.json` (register via `hokage-vision model register`)
- Weight path or release URL: `runs/closed-loop/train/run/weights/best.pt` (training) / `best.onnx` (inference export)
- SHA256: `best.pt` 231bde23ffd7cff0… / `best.onnx` ef8e74fab89c23cf…

## Intended Use

This model is intended for local research, portfolio demonstration, and controlled anime character detection experiments. It is not a public dataset redistribution mechanism and is not cleared for commercial use unless the training data and weight license explicitly allow it.

**Important scope note:** this model was trained purely on synthetic geometric shapes (colored rectangles), so it detects the synthetic color/shape signature used by the closed-loop demo — not real anime characters. It exists to prove the full generate → train → export → detect pipeline, not to perform real character recognition.

## Classes

- `obito` (blue-purple rectangle, class 0)
- `naruto` (orange rectangle, class 1)
- `gaara` (red rectangle, class 2)

## Data Provenance

- Dataset manifest: `data/manifests/hokage-vision-sample.yaml` (synthetic smoke family) / generated per-run by `scripts/closed_loop_demo.py`
- Image sources: programmatically generated rectangles on dark background with noise blocks (`PIL`, zero external data)
- Redistribution allowed: `true` (Apache-2.0, no third-party content)
- Annotation review status: reviewed (auto-generated from the drawing code, exact ground truth)
- Known excluded data: copyrighted screenshots or private captures without documented rights.

## Training Configuration

- Backend: `ultralytics`
- Base model: `yolov8n.pt` (COCO-pretrained, Ultralytics AGPL-3.0)
- Epochs: 40 (best at epoch 38)
- Batch size: 16
- Image size: 320
- Device: CPU
- Dataset: 3 classes × 40 train + 8 val synthetic images (120 total)
- Output directory: `runs/closed-loop/train/run/`
- Reproduce: `python scripts/closed_loop_demo.py --epochs 40 --figure-out docs/screenshots/closed-loop-detection.png`

## Evaluation

- Evaluation dataset: 24 synthetic val images (8 per class), top-1 label match
- Val accuracy: 24/24 (100%), all confidences ≈ 1.00
- mAP50: `0.995`
- mAP50-95: `0.995`
- Precision: `0.994`
- Recall: `1.000`
- Latency: ~0.1s/image (CPU, ONNX Runtime)
- Model size: 24.4 MB (`best.pt`) / 12.1 MB (`best.onnx`)

## Limitations

- Trained on synthetic colored rectangles only; it will NOT detect real anime characters, faces, or natural images.
- Single-instance scenes by construction (one rectangle per image); crowded-scene behavior is untested.
- Performance depends on reviewed, representative training images and bounding-box annotations.
- Adding a new character requires new data, new labels, updated class names, retraining, evaluation, and registry updates.
- The Agent can orchestrate training and evaluation, but it cannot create legal data or reliable new classes from nothing.

## License Notes

- Model weight license: inherits Ultralytics AGPL-3.0 (fine-tuned from `yolov8n.pt`)
- Training data license: Apache-2.0 (project-generated synthetic images)
- Annotation license: Apache-2.0 (auto-generated ground truth)
- Redistribution notes: synthetic dataset freely redistributable; the fine-tuned weights carry AGPL-3.0 obligations from the YOLOv8 base model and must not be treated as Apache-2.0 artifacts.
