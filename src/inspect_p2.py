from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11n_p2.yaml"


model = YOLO(str(MODEL_CONFIG))

detect = model.model.model[-1]

print("\n===== Detect Head =====")
print(detect)

print("\n===== Detection Strides =====")
print(detect.stride)

print("\n===== Number of Detection Scales =====")
print(len(detect.stride))