"""Person detector supporting both RF-DETR and YOLO (e.g. YOLO26n) on CPU, MPS, and CUDA."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import warnings

import numpy as np

VARIANTS = {
    "nano": "RFDETRNano", "rfdetr_nano": "RFDETRNano", "rfdetr-nano": "RFDETRNano", "rf-detr-nano": "RFDETRNano",
    "small": "RFDETRSmall", "rfdetr_small": "RFDETRSmall", "rfdetr-small": "RFDETRSmall", "rf-detr-small": "RFDETRSmall",
    "medium": "RFDETRMedium", "rfdetr_medium": "RFDETRMedium", "rfdetr-medium": "RFDETRMedium",
    "base": "RFDETRBase", "rfdetr_base": "RFDETRBase",
    "large": "RFDETRLarge", "rfdetr_large": "RFDETRLarge",
}
COCO_PERSON = 1

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YOLO26N_PATH = MODELS_DIR / "yolo26n.pt"
YOLO26S_PATH = MODELS_DIR / "yolo26s.pt"


class PersonDetector:
    """Returns `(boxes (N, 4) x1 y1 x2 y2, scores (N,))` for the people in an RGB frame, best first."""

    def __init__(self, variant: str = "yolo26s", confidence: float = 0.5, max_persons: int = 4,
                 device: str = "mps", use_pose_envelope: bool = False):
        self.confidence, self.max_persons = confidence, max_persons
        self.device = device
        self.use_pose_envelope = use_pose_envelope
        self.is_yolo = (
            str(variant).lower().startswith("yolo")
            or str(variant).endswith(".pt")
            or Path(variant).exists()
        )

        if self.is_yolo:
            os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
            os.makedirs("/tmp/ultralytics", exist_ok=True)
            from ultralytics import YOLO
            import torch

            # Resolve model weights path from local models/ dir first
            v_lower = str(variant).lower()
            if (MODELS_DIR / f"{v_lower}.pt").exists():
                weights = str(MODELS_DIR / f"{v_lower}.pt")
            elif (MODELS_DIR / variant).exists():
                weights = str(MODELS_DIR / variant)
            elif Path(variant).exists():
                weights = str(variant)
            else:
                # Check for yolo26s default in models/
                target_pt = MODELS_DIR / f"{v_lower}.pt" if v_lower.startswith("yolo") else (MODELS_DIR / "yolo26s.pt")
                if target_pt.exists():
                    weights = str(target_pt)
                else:
                    # Let Ultralytics resolve or download to models dir
                    weights = str(variant)

            # Resolve device: MPS if requested and available, else CPU/CUDA
            if device == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                self.device = "cpu"
            self.model = YOLO(weights)
        else:
            import rfdetr

            logging.getLogger("rf-detr").setLevel(logging.ERROR)
            v_key = str(variant).lower()
            if v_key not in VARIANTS:
                raise ValueError(f"unknown detector variant {variant!r}: choose one of {', '.join(VARIANTS)} or 'yolo26s'")
            self.model = getattr(rfdetr, VARIANTS[v_key])()
            if hasattr(self.model, "optimize_for_inference"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model.optimize_for_inference()

    def __call__(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.is_yolo:
            import cv2
            # Ultralytics YOLO assumes numpy array inputs are in BGR (OpenCV format).
            # Convert RGB to BGR to ensure true color representation and maximum detector accuracy.
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if (image_rgb.ndim == 3 and image_rgb.shape[2] == 3) else image_rgb
            results = self.model.predict(
                img_bgr,
                device=self.device,
                conf=self.confidence,
                verbose=False
            )[0]
            if len(results.boxes) == 0:
                return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)
            
            # For YOLO-Pose, person is class 0. For standard COCO detection, person is class 0.
            cls = results.boxes.cls.cpu().numpy()
            keep = (cls == 0)
            boxes = results.boxes.xyxy.cpu().numpy()[keep].astype(np.float32)
            scores = results.boxes.conf.cpu().numpy()[keep].astype(np.float32)

            if self.use_pose_envelope and hasattr(results, "keypoints") and results.keypoints is not None:
                kps = results.keypoints.xy.cpu().numpy()[keep]  # (N, 17, 2)
                h_img, w_img = image_rgb.shape[:2]
                for i in range(len(boxes)):
                    pkps = kps[i]
                    valid = pkps[(pkps[:, 0] > 0) | (pkps[:, 1] > 0)]
                    if len(valid) > 0:
                        k_min = valid.min(axis=0)
                        k_max = valid.max(axis=0)
                        # Ensure box encompasses all outstretched limbs (e.g. wrists/elbows in bench press)
                        pad = 12.0
                        boxes[i, 0] = max(0.0, min(boxes[i, 0], float(k_min[0] - pad)))
                        boxes[i, 1] = max(0.0, min(boxes[i, 1], float(k_min[1] - pad)))
                        boxes[i, 2] = min(float(w_img), max(boxes[i, 2], float(k_max[0] + pad)))
                        boxes[i, 3] = min(float(h_img), max(boxes[i, 3], float(k_max[1] + pad)))

            order = np.argsort(-scores)[: self.max_persons]
            return boxes[order], scores[order]
        else:
            from PIL import Image

            det = self.model.predict(Image.fromarray(image_rgb), threshold=self.confidence)
            keep = det.class_id == COCO_PERSON if det.class_id is not None else np.zeros(len(det), bool)
            boxes, scores = det.xyxy[keep].astype(np.float32), det.confidence[keep].astype(np.float32)
            order = np.argsort(-scores)[: self.max_persons]
            return boxes[order], scores[order]
