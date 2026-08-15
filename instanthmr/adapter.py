"""
InstantHMR Landmark Adapter
===========================
Self-contained module for cloud/backend instances.
Runs InstantHMR (YOLOv8n + InstantHMR ONNX) and outputs exact MediaPipe / MLKit
landmark dictionary format for downstream biomechanics pipelines.
"""

import os
import sys
import time
import cv2
import numpy as np

# Resolve repo root directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from instanthmr.pipeline import PosePipeline

# MHR 70-joint index to MediaPipe landmark name mapping
MAPPING_MHR_TO_MP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_heel": 17,
    "right_heel": 18,
    "left_foot_index": 19,
    "right_foot_index": 20,
    "left_pinky": 38,
    "left_index": 26,
    "right_pinky": 59,
    "right_index": 47,
    "left_mouth": 21,
    "right_mouth": 22,
}

_GLOBAL_PIPELINE = None

def get_pipeline(
    onnx_path: str = None,
    detector_variant: str = None,
    device: str = "cpu",
    detector_stride: int = 3
) -> PosePipeline:
    """Get or initialize a cached singleton PosePipeline instance."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        if onnx_path is None:
            onnx_path = os.path.join(REPO_ROOT, "models", "instanthmr.onnx")
        if detector_variant is None:
            # Check models/ folder first, fallback to root or download
            model_p = os.path.join(REPO_ROOT, "models", "yolov8n.pt")
            detector_variant = model_p if os.path.exists(model_p) else "yolov8n.pt"

        _GLOBAL_PIPELINE = PosePipeline(
            onnx_path=onnx_path,
            device=device,
            detector_type="yolo",
            detector_variant=detector_variant,
            det_confidence=0.35,
            max_persons=1,
            detector_stride=detector_stride,
            batch_persons=True,
        )
    return _GLOBAL_PIPELINE


def process_video_with_instanthmr(video_path: str, device: str = "cpu", detector_stride: int = 3):
    """
    Drop-in replacement for process_video_with_mediapipe.

    Args:
        video_path: Path to the local MP4/MOV video file.
        device: 'cpu' or 'cuda'.
        detector_stride: YOLO execution frequency (default 3).

    Returns:
        landmark_dict: Dict of landmark labels -> list of dicts with keys ('t', 'x', 'y', 'z', 'label')
        frame_time: Delta time per frame (1 / fps)
    """
    pipeline = get_pipeline(device=device, detector_stride=detector_stride)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video at: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time = 1.0 / fps

    landmark_dict = {label: [] for label in MAPPING_MHR_TO_MP.keys()}
    frame_index = 0
    start_timestamp = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = pipeline.predict(frame_rgb)
        timestamp = start_timestamp + (frame_index * frame_time)

        if result.persons:
            person = result.persons[0]
            j2d = person.joints_2d        # (70, 2) image pixel coordinates
            j3d_cam = person.joints_3d_cam # (70, 3) metric camera space

            for label, joint_idx in MAPPING_MHR_TO_MP.items():
                x_pixel = int(j2d[joint_idx, 0])
                y_pixel = int(j2d[joint_idx, 1])
                z_real = float(j3d_cam[joint_idx, 2] * width)

                landmark_dict[label].append({
                    "t": timestamp,
                    "x": y_pixel,
                    "y": -x_pixel,
                    "z": z_real,
                    "label": label
                })
        else:
            for label in MAPPING_MHR_TO_MP.keys():
                if landmark_dict[label]:
                    prev = landmark_dict[label][-1].copy()
                    prev["t"] = timestamp
                    landmark_dict[label].append(prev)

        frame_index += 1

    cap.release()
    return landmark_dict, frame_time
