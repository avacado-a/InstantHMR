"""
InstantHMR Landmark Adapter
===========================
Self-contained module for cloud/backend instances.
Runs InstantHMR (YOLOv8n + InstantHMR ONNX) and outputs exact MediaPipe / MLKit
landmark dictionary format for downstream biomechanics pipelines.
Includes full 33-landmark coverage (70 body joints + LOD 6 MHR mesh mouth & eye vertices).
"""

import os
import sys
import time
import math
import cv2
import numpy as np

# Resolve repo root directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from instanthmr.pipeline import PosePipeline
from instanthmr.mhr_renderer import MHRRenderer

# 1. 27 Core Body & Limb Landmarks (MHR 70-joint index mapping)
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
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "left_foot_index": 15,     # left_big_toe_tip
    "right_foot_index": 18,    # right_big_toe_tip
    "left_heel": 17,
    "right_heel": 20,
    "right_thumb": 21,         # right_thumb_tip
    "right_index": 25,         # right_index_tip
    "right_pinky": 37,         # right_pinky_tip
    "right_wrist": 41,         # right_wrist
    "left_thumb": 42,          # left_thumb_tip
    "left_index": 46,          # left_index_tip
    "left_pinky": 58,          # left_pinky_tip
    "left_wrist": 62,          # left_wrist
}

# 2. 6 Auxiliary Facial Detail Landmarks (MHR LOD 6 Mesh Vertex index mapping)
MAPPING_MESH_VERT_TO_MP = {
    "mouth_left": 47,
    "mouth_right": 350,
    "left_eye_inner": 41,
    "left_eye_outer": 19,
    "right_eye_inner": 319,
    "right_eye_outer": 321,
}

ALL_MP_LANDMARK_LABELS = list(MAPPING_MHR_TO_MP.keys()) + list(MAPPING_MESH_VERT_TO_MP.keys())

_GLOBAL_PIPELINE = None
_GLOBAL_MHR_RENDERER = None

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


def get_mhr_renderer(assets_folder: str = None, device: str = "cpu", lod: int = 6) -> MHRRenderer:
    """Get or initialize cached singleton MHRRenderer instance for mesh vertex extraction.
    Automatically downloads and extracts official MHR assets if not present.
    """
    global _GLOBAL_MHR_RENDERER
    if _GLOBAL_MHR_RENDERER is None:
        if assets_folder is None:
            assets_folder = os.path.join(REPO_ROOT, "assets")
            if not os.path.exists(assets_folder):
                assets_folder = os.path.join(REPO_ROOT, "models", "mhr_assets")

        # Auto-download from Meta's official release if missing on clean cloud setup
        mhr_pt_path = os.path.join(assets_folder, "mhr_model.pt")
        if not os.path.exists(mhr_pt_path):
            import urllib.request
            import zipfile
            print("MHR assets not found locally. Downloading official assets from Meta release...")
            os.makedirs(assets_folder, exist_ok=True)
            zip_path = os.path.join(REPO_ROOT, "assets_tmp.zip")
            url = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(assets_folder)
            os.remove(zip_path)
            # Remove heavy LOD 0-5 to preserve disk space
            for i in range(6):
                for fn in [f"corrective_blendshapes_lod{i}.npz", f"lod{i}.fbx"]:
                    p = os.path.join(assets_folder, fn)
                    if os.path.exists(p):
                        os.remove(p)
            print("MHR assets downloaded and configured for LOD 6.")

        # Hard-locked strictly to CPU:
        _GLOBAL_MHR_RENDERER = MHRRenderer(assets_folder=assets_folder, device="cpu", lod=lod)
    return _GLOBAL_MHR_RENDERER


def process_video_with_instanthmr(
    video_path: str,
    device: str = "cpu",
    detector_stride: int = 3,
    include_face_mesh: bool = True
):
    """
    Drop-in replacement for process_video_with_mediapipe.
    Outputs full MediaPipe/MLKit landmark dictionary format.

    Args:
        video_path: Path to the local MP4/MOV video file.
        device: 'cpu' or 'cuda'.
        detector_stride: YOLO execution frequency (default 3).
        include_face_mesh: Whether to decode LOD 6 mesh for exact mouth and eye corners.

    Returns:
        landmark_dict: Dict of landmark labels -> list of dicts with keys ('t', 'x', 'y', 'z', 'label')
        frame_time: Delta time per frame (1 / fps)
    """
    pipeline = get_pipeline(device=device, detector_stride=detector_stride)
    mhr_renderer = get_mhr_renderer(device=device, lod=6) if include_face_mesh else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video at: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_time = 1.0 / fps

    target_labels = ALL_MP_LANDMARK_LABELS if include_face_mesh else list(MAPPING_MHR_TO_MP.keys())
    landmark_dict = {label: [] for label in target_labels}
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

            # 1. Map 27 Core Body Joints
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

            # 2. Extract 6 Auxiliary Facial Landmarks with Eye-Center Alignment
            if include_face_mesh and mhr_renderer is not None:
                # Forward pass for LOD 6 mesh vertices in body-local coordinates
                verts_local = mhr_renderer.forward(person.mhr_params, person.shape_params)  # (595, 3)
                
                # Transform to camera space
                verts_cam = verts_local + person.cam_trans  # (595, 3)

                # Pinhole projection for all mesh vertices
                f = person.focal_length  # [fx, fy]
                cx, cy = person.principal_point[0], person.principal_point[1]

                raw_mesh_2d = {}
                for label, v_idx in MAPPING_MESH_VERT_TO_MP.items():
                    vc = verts_cam[v_idx]
                    zc = max(1e-3, float(vc[2]))
                    u = (vc[0] * f[0] / zc) + cx
                    v = (vc[1] * f[1] / zc) + cy
                    raw_mesh_2d[label] = np.array([u, v], dtype=np.float32)

                # Eye Center Matching:
                # Left eye center from model: j2d[1] (left_eye)
                # Left eye center from mesh: midpoint of left_eye_inner (41) and left_eye_outer (19)
                mesh_l_eye_center = (raw_mesh_2d["left_eye_inner"] + raw_mesh_2d["left_eye_outer"]) / 2.0
                offset_l = j2d[1] - mesh_l_eye_center

                # Right eye center from model: j2d[2] (right_eye)
                # Right eye center from mesh: midpoint of right_eye_inner (319) and right_eye_outer (321)
                mesh_r_eye_center = (raw_mesh_2d["right_eye_inner"] + raw_mesh_2d["right_eye_outer"]) / 2.0
                offset_r = j2d[2] - mesh_r_eye_center

                # Average facial rigid alignment delta
                face_offset = (offset_l + offset_r) / 2.0

                for label, vert_idx in MAPPING_MESH_VERT_TO_MP.items():
                    v_cam = verts_cam[vert_idx]
                    z_cam = max(1e-3, float(v_cam[2]))

                    if "left_eye" in label:
                        pt_2d = raw_mesh_2d[label] + offset_l
                    elif "right_eye" in label:
                        pt_2d = raw_mesh_2d[label] + offset_r
                    else:
                        # Mouth corners follow face rigid offset
                        pt_2d = raw_mesh_2d[label] + face_offset

                    px_2d = int(pt_2d[0])
                    py_2d = int(pt_2d[1])
                    pz_real = float(z_cam * width)

                    landmark_dict[label].append({
                        "t": timestamp,
                        "x": py_2d,
                        "y": -px_2d,
                        "z": pz_real,
                        "label": label
                    })
        else:
            for label in target_labels:
                if landmark_dict[label]:
                    prev = landmark_dict[label][-1].copy()
                    prev["t"] = timestamp
                    landmark_dict[label].append(prev)

        frame_index += 1

    cap.release()
    return landmark_dict, frame_time
