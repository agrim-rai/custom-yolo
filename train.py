from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8n.yaml")
    model.train(
        data=ROOT / "ultralytics" / "cfg" / "datasets" / "african-wildlife.yaml",
        epochs=10,
        imgsz=64
    )