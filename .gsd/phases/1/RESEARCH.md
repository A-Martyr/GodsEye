# Research: Deep Learning OCR Upgrade (Phase 1)

## Context
Currently, npr/ocr.py uses cv2.morphologyEx to detect plate candidates. This fails under poor lighting or severe angles. The objective is to swap this seam out for a deep learning model.

## Options
1. **OpenCV DNN Module with MobileNet SSD:**
   - Pros: Built-in to current dependencies (opencv-python), lightweight.
   - Cons: Harder to find good pre-trained weights for modern license plates without manual training.
2. **Ultralytics YOLO (YOLOv8):**
   - Pros: State-of-the-art accuracy, very easy to use Python API (model.predict()), pre-trained license plate models are widely available or easily fine-tuned.
   - Cons: Requires adding ultralytics and 	orch to dependencies.

## Decision
Go with **YOLOv8 (Ultralytics)**. It perfectly aligns with the 'enterprise-grade' requirement and achieving >90% accuracy in real-world conditions.

## Implementation Details
1. Add ultralytics to equirements.txt.
2. Update npr/ocr.py's detect_plate_candidates function.
3. Instead of morphological operations, initialize a YOLO model (stubbed as 'yolov8n-license-plate.pt' or similar fallback) and return the bounding boxes.
4. Convert YOLO bounds to the (int, int, int, int) format currently expected by the downstream reader.
