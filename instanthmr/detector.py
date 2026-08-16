"""YOLO Person Detector for InstantHMR.

Wraps Ultralytics YOLO (e.g. yolov8n.pt or ONNX) to produce person bounding
boxes for upstream InstantHMR pose inference.
"""

from __future__ import annotations
import numpy as np

COCO_PERSON_CLASS_ID = 0


class YOLODetector:
    """Person detector backed by Ultralytics YOLO.

    Args:
        model_name: YOLO model weights or ONNX path (default: ``"yolov8n.pt"``).
        confidence: Minimum detection score.
        max_persons: Maximum number of persons returned per frame.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        max_persons: int = 1,
    ):
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._confidence = confidence
        self._max_persons = max_persons
        self._variant = model_name

    def warmup(self) -> None:
        """Run one silent forward pass."""
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self.detect(dummy)

    @property
    def variant(self) -> str:
        return self._variant

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Detect persons in an RGB image using YOLO."""
        results = self._model.predict(
            image_rgb,
            conf=self._confidence,
            max_det=self._max_persons,
            classes=[COCO_PERSON_CLASS_ID],  # COCO class 0 is person
            verbose=False,
        )

        detections: list[dict] = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for bbox, conf in zip(xyxy, confs):
                detections.append({
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "confidence": float(conf),
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[: self._max_persons]
