---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Integrate YOLO Object Detection for Plate Localization

## Objective
Replace the classical morphological plate localization algorithm with a trained YOLO model to drastically improve detection accuracy in diverse real-world conditions.

## Context
- .gsd/SPEC.md
- .gsd/phases/1/RESEARCH.md
- anpr/ocr.py
- requirements.txt

## Tasks

<task type="auto">
  <name>Update dependencies</name>
  <files>requirements.txt</files>
  <action>
    Add ultralytics>=8.0.0 to the # vision + OCR section of equirements.txt.
  </action>
  <verify>python -c "print('ultralytics' in open('requirements.txt').read())"</verify>
  <done>ultralytics is present in requirements.txt.</done>
</task>

<task type="auto">
  <name>Upgrade detect_plate_candidates</name>
  <files>anpr/ocr.py</files>
  <action>
    - Import YOLO from ultralytics in npr/ocr.py (inside a try-except to handle environments where it's not installed yet).
    - Modify detect_plate_candidates(frame, max_candidates=12) to run inference using a YOLO model (default to a placeholder 'plate_detector.pt').
    - Fallback gracefully to the old morphological approach if the model file is not found on disk, ensuring the system doesn't break in environments without the weights.
    - Extract bounding boxes from the YOLO results and return them in the expected format (bounding boxes and scores). The current format is list[tuple[int, int, int, int]]. Actually, the current function returns boxes.
  </action>
  <verify>python -c "import anpr.ocr; import numpy as np; anpr.ocr.detect_plate_candidates(np.zeros((400, 400, 3), dtype=np.uint8))"</verify>
  <done>The function runs without crashing and falls back to morphology if model weights are missing.</done>
</task>

## Success Criteria
- [ ] ultralytics is added to equirements.txt.
- [ ] detect_plate_candidates supports YOLO when available.
- [ ] Downstream compatibility is preserved.
