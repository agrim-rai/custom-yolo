# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# detection-only: YOLO object detection

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """YOLO object detection model (detection-only build)."""

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        path = Path(model if isinstance(model, (str, Path)) else "")
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }
