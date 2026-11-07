"""Person detection followed by InstantHMR, with per-stage timings."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import PersonDetector
from .inference import HMRPrediction, InstantHMR
from .models import DEFAULT_MODEL, resolve_model


@dataclass
class FrameResult:
    persons: list[HMRPrediction]
    detector_ms: float
    hmr_ms: float


class PosePipeline:
    """Person detector (YOLO / RF-DETR), then InstantHMR on every detected person in one batched call.

    `model` is a released model name (nano, tiny, small, medium, medium-commercial) or an .onnx
    path. `device` can be "mps", "cpu", "coreml", "cuda", or "tensorrt".
    `detector_stride` > 1 runs the detector every N frames and reuses its boxes in between.
    `with_mesh`=False skips the heavy 18,439-vertex MHR mesh calculation when only keypoints are needed.
    """

    def __init__(self, model: str | Path = DEFAULT_MODEL, device: str = "mps", detector: str = "yolo26n",
                 det_confidence: float = 0.5, max_persons: int = 4, detector_stride: int = 4,
                 focal: float | None = None, detector_device: str | None = None,
                 with_mesh: bool = True, crop_expand: float = 1.4, use_pose_envelope: bool = False):
        det_dev = detector_device or ("mps" if device in ("mps", "coreml") else device)
        self.hmr = InstantHMR(resolve_model(model), device=device, focal=focal, max_batch=max_persons,
                              with_mesh=with_mesh, crop_expand=crop_expand)
        self.detector = PersonDetector(detector, confidence=det_confidence, max_persons=max_persons,
                                       device=det_dev, use_pose_envelope=use_pose_envelope)
        self.detector_stride = max(1, detector_stride)
        self.with_mesh = with_mesh
        self.crop_expand = crop_expand
        self.use_pose_envelope = use_pose_envelope
        self._frame = 0
        self._boxes = np.zeros((0, 4), np.float32)
        self._scores = np.zeros(0, np.float32)
        self.detector(np.zeros((288, 288, 3), np.uint8))
        self.hmr.warmup(max_persons)

    def __call__(self, image_rgb: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        if self._frame % self.detector_stride == 0 or len(self._boxes) == 0:
            self._boxes, self._scores = self.detector(image_rgb)
        t1 = time.perf_counter()
        persons = self.hmr.predict(image_rgb, self._boxes, self._scores)
        t2 = time.perf_counter()
        self._frame += 1
        return FrameResult(persons, (t1 - t0) * 1000, (t2 - t1) * 1000)
